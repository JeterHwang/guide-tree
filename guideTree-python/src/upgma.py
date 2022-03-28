from os import sep
from matplotlib.pyplot import axis
import numpy as np
from typing import Dict, List
from .utils import distMatrix
BIG_DIST = 1e29

__all__ = [
    'TreeNode',
    'UPGMA',
]

class TreeNode:
    def __init__(
        self, 
        isLeaf, 
        ID,
        data, 
        leftIndex=None, 
        rightIndex=None,
        height=0,
        rightlength=0,
        leftlength=0
    ) -> None:
        self.isLeaf = isLeaf
        self.ID = ID
        self.data = data if isLeaf is True else None
        self.left = leftIndex
        self.right = rightIndex
        
        self.height = height
        self.rightLength = rightlength
        self.leftLength = leftlength
    
    def getRight(self):
        return self.left
    def getLeft(self):
        return self.right
    def getName(self):
        return self.data['name'] if self.data is not None else f"Internal Node ({self.ID})"

###### BROKEN UPGMA ######
class UPGMA:
    def __init__(
        self, 
        seqs : List[Dict],
        linkage_type : str,
        dist_type : str,
    ) -> None:
        
        self.distmat = distMatrix(seqs, dist_type)
        self.linkage_type = linkage_type
        
        self.leafNodeCount = self.distmat.shape[0]
        self.internalNodeCount = self.leafNodeCount - 1

        self.Tree : List[TreeNode] = []
        for i in range(self.leafNodeCount):
            self.Tree.append(TreeNode(True, i, seqs[i]))

        self.mapping : np.ndarray = np.arange(self.leafNodeCount)
        self.NN : np.ndarray = np.argmin(self.distmat, axis=1)
        self.minDist : np.ndarray = np.amin(self.distmat, axis=1)

        self.cluter()
        self.DFS(self.Tree[-1])

    def cluster_one_iteration(self):
        # print(self.NN)
        # print(self.minDist)
        # print(self.distmat)
        Lmin = np.argmin(self.minDist)
        Rmin = self.NN[Lmin]
        newDist : float 
        newMinDist : float = BIG_DIST
        newNN : int = -1

        for i in range(self.leafNodeCount):
            if i == Lmin or i == Rmin or self.mapping[i] == -1:
                continue
            
            if self.linkage_type == 'AVG':
                newDist = (self.distmat[Lmin][i] + self.distmat[Rmin][i]) / 2
                # print(self.distmat[Lmin][i], self.distmat[Rmin][i], newDist, sep=' ')
            elif self.linkage_type == 'MIN':
                newDist = min(self.distmat[Lmin][i], self.distmat[Rmin][i])
            elif self.linkage_type == 'MAX':
                newDist = max(self.distmat[Lmin][i], self.distmat[Rmin][i])
            else:
                raise NotImplementedError

            self.distmat[Lmin][i] = newDist
            self.distmat[i][Lmin] = newDist
            
            ## assume C(i,j) = NN(K) <=> C(i) = NN(K) or C(j) = NN(K)
            if self.NN[i] == Rmin:
                # print(f'update {i}, new minDist = {newDist}')
                # if i == 2:
                #     print(self.distmat[Lmin][i], self.distmat[Rmin][i], newDist, sep=' ')
                self.NN[i] = Lmin
                # self.minDist[i] = newDist

            if newDist < newMinDist:
                newMinDist = newDist
                newNN = i

        # Add New Node
        newID = len(self.Tree)
        dLR = self.distmat[Lmin][Rmin]
        new_height = dLR / 2
        Lindex = self.mapping[Lmin]
        Rindex = self.mapping[Rmin]
        Llength = new_height - self.Tree[Lindex].height
        Rlength = new_height - self.Tree[Rindex].height
        # print(newID, Lindex, Rindex, new_height)
        self.Tree.append(TreeNode(False, newID, None, Lindex, Rindex, new_height, Rlength, Llength))

        # Update New Node
        self.mapping[Lmin] = newID
        self.minDist[Lmin] = newMinDist
        self.NN[Lmin] = newNN
        
        # Delete Node
        self.mapping[Rmin] = -1
        self.minDist[Rmin] = BIG_DIST
        self.NN[Rmin] = -1

    def cluter(self):
        for iter in range(self.internalNodeCount):
            print(f"------ iteration {iter} ------")
            self.cluster_one_iteration()
    
    def writeTree(self, file):
        pass
    
    def appendTree(self, tree):
        pass

    def DFS(self, root : TreeNode):
        if root.isLeaf:
            print(root.getName())
            return
        
        print(f"Start of {root.getName()}", root.height, root.leftLength, root.rightLength, sep=' ')
        L, R = root.left, root.right
        self.DFS(self.Tree[L])
        self.DFS(self.Tree[R])
        print(f"End of {root.getName()}")