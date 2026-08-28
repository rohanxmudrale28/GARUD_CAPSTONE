from __future__ import annotations
import json, os, time, threading
from collections import deque
from pathlib import Path
import cv2
from flask import Flask, Response, jsonify, render_template, request, send_from_directory
from ultralytics import YOLO
from traffic import TrafficViolationEngine
from traffic.signal_controller import SignalController

ROOT=Path(__file__).resolve().parent
app=Flask(__name__,template_folder='templates',static_folder='static')
CAMERAS=json.loads((ROOT/'config_data/cameras.json').read_text())
EVENTS=deque(maxlen=500); LOCK=threading.Lock(); PROCESSORS={}

class CameraProcessor:
    def __init__(self,cid,cfg):
        self.cid=cid; self.cfg=cfg; self.frame=None; self.online=False; self.fps=0.0
        self.counts={'WRONG_WAY':0,'RED_LIGHT_JUMP':0,'RESTRICTED_LANE':0,'NO_ENTRY':0,'ILLEGAL_STOPPING':0}
        self.signal=SignalController(); self.engine=TrafficViolationEngine(cid,cfg)
        self.model=YOLO(str(ROOT/'yolov8n.pt') if (ROOT/'yolov8n.pt').exists() else 'yolov8n.pt')
        threading.Thread(target=self.run,daemon=True).start()
    def run(self):
        src=self.cfg.get('source',0); cap=cv2.VideoCapture(src); last=time.time()
        while True:
            ok,frame=cap.read()
            if not ok:
                self.online=False
                if isinstance(src,str): cap.set(cv2.CAP_PROP_POS_FRAMES,0); time.sleep(.2); continue
                time.sleep(1); continue
            self.online=True; state=self.signal.update()
            result=self.model.track(frame,persist=True,classes=[1,2,3,5,7],conf=.35,iou=.5,verbose=False)[0]
            detections=[]
            if result.boxes is not None and result.boxes.id is not None:
                for box,tid,cls,conf in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.id.int().cpu().tolist(),result.boxes.cls.int().cpu().tolist(),result.boxes.conf.cpu().tolist()):
                    detections.append({'bbox':box,'track_id':tid,'class_id':cls,'class_name':result.names[cls],'confidence':conf})
                    x1,y1,x2,y2=map(int,box); cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,255),2); cv2.putText(frame,f'{result.names[cls]} #{tid}',(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2)
            new=self.engine.process(detections,frame.shape,state)
            for e in new:
                name=f'{self.cid}_{int(e.timestamp)}_{e.event_type}.jpg'; path=ROOT/'snapshots'/name; cv2.imwrite(str(path),frame); e.snapshot=f'/snapshots/{name}'
                with LOCK: EVENTS.appendleft(e.to_dict()); self.counts[e.event_type]=self.counts.get(e.event_type,0)+1
            self.engine.draw_rules(frame,state)
            cv2.putText(frame,f'SIGNAL: {state.upper()}',(20,35),cv2.FONT_HERSHEY_SIMPLEX,.85,(0,0,255) if state=='red' else (0,255,0),2)
            now=time.time(); self.fps=.9*self.fps+.1/(max(now-last,.001)); last=now
            cv2.putText(frame,f'FPS {self.fps:.1f}',(20,68),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,255),2)
            ok,jpg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,82])
            if ok: self.frame=jpg.tobytes()
    def stream(self):
        while True:
            if self.frame: yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'+self.frame+b'\r\n'
            time.sleep(.04)

for cid,cfg in CAMERAS.items(): PROCESSORS[cid]=CameraProcessor(cid,cfg)

@app.get('/')
def home(): return render_template('dashboard.html')
@app.get('/api/cameras')
def cameras():
    data=[]
    for cid,cfg in CAMERAS.items():
        p=PROCESSORS[cid]; recent=next((e for e in EVENTS if e['camera_id']==cid),None)
        severity=recent['severity'] if recent and time.time()-recent['timestamp']<30 else 'clear'
        data.append({'id':cid,**{k:cfg[k] for k in ('name','city','lat','lng')},'online':p.online,'fps':round(p.fps,1),'signal':p.signal.state,'severity':severity,'counts':p.counts})
    return jsonify(data)
@app.get('/api/events')
def events(): return jsonify(list(EVENTS)[:int(request.args.get('limit',50))])
@app.post('/api/signal/<cid>')
def signal(cid):
    PROCESSORS[cid].signal.set((request.get_json(silent=True) or {}).get('state','')); return jsonify({'state':PROCESSORS[cid].signal.state})
@app.get('/api/stream/<cid>')
def stream(cid): return Response(PROCESSORS[cid].stream(),mimetype='multipart/x-mixed-replace; boundary=frame')
@app.get('/snapshots/<path:name>')
def snapshots(name): return send_from_directory(ROOT/'snapshots',name)

if __name__=='__main__': app.run(host='0.0.0.0',port=int(os.getenv('PORT','5001')),threaded=True)
