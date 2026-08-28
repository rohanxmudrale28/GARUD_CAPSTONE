import time
class SignalController:
    def __init__(self, red=15, green=15, amber=3):
        self.durations={'red':red,'green':green,'amber':amber}; self.state='red'; self.changed=time.time()
    def update(self):
        if time.time()-self.changed >= self.durations[self.state]:
            self.state={'red':'green','green':'amber','amber':'red'}[self.state]; self.changed=time.time()
        return self.state
    def set(self,state):
        if state in self.durations: self.state=state; self.changed=time.time()
