from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import time, math, uuid
import cv2
import numpy as np

Point = Tuple[int, int]

@dataclass
class TrafficEvent:
    id: str
    camera_id: str
    event_type: str
    severity: str
    track_id: int
    timestamp: float
    confidence: float
    message: str
    snapshot: Optional[str] = None
    def to_dict(self): return asdict(self)

def side(a: Point, b: Point, p: Point) -> float:
    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])

def crossed(prev: Point, curr: Point, a: Point, b: Point) -> bool:
    return side(a,b,prev) * side(a,b,curr) < 0

def in_poly(p: Point, poly: List[Point]) -> bool:
    return cv2.pointPolygonTest(np.asarray(poly, np.int32), p, False) >= 0

def unit(v):
    n = math.hypot(v[0], v[1]) or 1.0
    return v[0]/n, v[1]/n

class TrafficViolationEngine:
    """Rule engine operating on stable YOLO track IDs.

    Camera config coordinates are normalized from 0..1 and converted per frame.
    Required detector classes: car, motorcycle, bus, truck, bicycle.
    """
    VEHICLES = {1, 2, 3, 5, 7}  # COCO: bicycle, car, motorcycle, bus, truck

    def __init__(self, camera_id: str, config: dict, cooldown_seconds: float = 12):
        self.camera_id = camera_id
        self.cfg = config
        self.cooldown = cooldown_seconds
        self.history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
        self.last_alert: Dict[Tuple[int,str], float] = {}
        self.zone_entered: Dict[Tuple[int,str], float] = {}

    def _px_point(self, p, w, h): return int(p[0]*w), int(p[1]*h)
    def _px_poly(self, poly, w, h): return [self._px_point(p,w,h) for p in poly]

    def process(self, detections: List[dict], frame_shape, signal_state: str) -> List[TrafficEvent]:
        h, w = frame_shape[:2]
        now = time.time(); events = []
        for d in detections:
            tid = int(d['track_id']); cls = int(d['class_id'])
            if cls not in self.VEHICLES: continue
            x1,y1,x2,y2 = map(int, d['bbox'])
            c = ((x1+x2)//2, y2)  # road contact point
            hist = self.history[tid]
            prev = hist[-1][1] if hist else None
            hist.append((now,c))
            if prev:
                events += self._wrong_way(tid, prev, c, w, h)
                events += self._line_rules(tid, prev, c, w, h, signal_state)
            events += self._restricted_lane(tid, c, w, h, d.get('class_name','vehicle'))
            events += self._stopped_vehicle(tid, now, w, h)
        return events

    def _emit(self, tid, typ, severity, confidence, message):
        key=(tid,typ); now=time.time()
        if now-self.last_alert.get(key,0) < self.cooldown: return []
        self.last_alert[key]=now
        return [TrafficEvent(str(uuid.uuid4()), self.camera_id, typ, severity, tid, now, confidence, message)]

    def _wrong_way(self, tid, prev, curr, w, h):
        lane=self.cfg.get('wrong_way_lane');
        if not lane: return []
        poly=self._px_poly(lane['polygon'],w,h)
        if not in_poly(curr,poly): return []
        dx,dy=curr[0]-prev[0],curr[1]-prev[1]
        if math.hypot(dx,dy) < max(2, w*0.002): return []
        actual=unit((dx,dy)); allowed=unit(tuple(lane['allowed_direction']))
        dot=actual[0]*allowed[0]+actual[1]*allowed[1]
        if dot < float(lane.get('opposite_dot_threshold',-0.35)):
            return self._emit(tid,'WRONG_WAY','red',min(0.99,abs(dot)),f'Vehicle {tid} moving against permitted direction')
        return []

    def _line_rules(self, tid, prev, curr, w, h, signal):
        out=[]
        stop=self.cfg.get('stop_line')
        if stop and signal.lower()=='red':
            a=self._px_point(stop[0],w,h); b=self._px_point(stop[1],w,h)
            if crossed(prev,curr,a,b):
                out += self._emit(tid,'RED_LIGHT_JUMP','red',0.94,f'Vehicle {tid} crossed stop line during red signal')
        one=self.cfg.get('no_entry_line')
        if one:
            a=self._px_point(one[0],w,h); b=self._px_point(one[1],w,h)
            if crossed(prev,curr,a,b): out += self._emit(tid,'NO_ENTRY','amber',0.90,f'Vehicle {tid} crossed no-entry boundary')
        return out

    def _restricted_lane(self, tid, c, w, h, name):
        z=self.cfg.get('restricted_lane')
        if z and in_poly(c,self._px_poly(z['polygon'],w,h)) and name not in z.get('allowed_classes',[]):
            return self._emit(tid,'RESTRICTED_LANE','yellow',0.88,f'{name.title()} {tid} entered restricted lane')
        return []

    def _stopped_vehicle(self, tid, now, w, h):
        z=self.cfg.get('no_stopping_zone'); hist=self.history[tid]
        if not z or len(hist)<2 or not in_poly(hist[-1][1],self._px_poly(z['polygon'],w,h)): return []
        window=float(z.get('seconds',10)); recent=[p for t,p in hist if now-t<=window]
        if len(recent)>5:
            spread=max(math.hypot(p[0]-recent[0][0],p[1]-recent[0][1]) for p in recent)
            if spread < w*0.015 and hist[-1][0]-hist[0][0]>=window*0.8:
                return self._emit(tid,'ILLEGAL_STOPPING','yellow',0.84,f'Vehicle {tid} stopped in prohibited zone')
        return []

    def draw_rules(self, frame, signal_state):
        h,w=frame.shape[:2]
        for key,color in [('wrong_way_lane',(255,180,0)),('restricted_lane',(0,220,255)),('no_stopping_zone',(0,120,255))]:
            z=self.cfg.get(key)
            if z:
                pts=np.asarray(self._px_poly(z['polygon'],w,h),np.int32)
                cv2.polylines(frame,[pts],True,color,2)
        if self.cfg.get('stop_line'):
            a,b=[self._px_point(p,w,h) for p in self.cfg['stop_line']]
            cv2.line(frame,a,b,(0,0,255) if signal_state=='red' else (0,255,0),3)
        return frame
