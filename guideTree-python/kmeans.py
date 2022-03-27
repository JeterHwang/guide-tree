
import numpy as np
from typing import Dict, List
from kmeans_pytorch import kmeans, kmeans_predict

def Euclidean(P1 : Dict, P2 : Dict) -> float:
    return np.dot(P1.embedding - P2.embedding, P1.embedding - P2.embedding)

def BisectingKmeans(seqs : List[Dict]):
    pass


    