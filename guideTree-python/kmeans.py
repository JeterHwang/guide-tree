from math import log2
import numpy as np
from typing import Dict

class KmeansResult:
    def __init__(self, points : Dict) -> None:
        self.points = points
    
    def Kmeans(seqs : np.ndarray) -> Dict:
        pass

    @property
    def seedNumber(self):
        return int(log2(self.points.shape[0])**2)
    
    def sortbylen(self):
        
    