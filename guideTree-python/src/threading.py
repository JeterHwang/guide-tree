from concurrent.futures import thread
import time
import threading

class Worker(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(Worker, self).__init__(*args, **kwargs)
        self._return = None
    
    def run(self):
        if self._Thread__target is not None:
            self._return = self._Thread__target(*self._Thread__args, **self._Thread__kwargs)

    def join(self):
        super(Worker, self).join()
        return self._return