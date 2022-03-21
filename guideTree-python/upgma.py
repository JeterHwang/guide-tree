import numpy as np
from typing import List

BIG_DIST = 1e29

class TreeNode:
    def __init__(
        self, 
        isLeaf, 
        rootName, 
        leftIndex=None, 
        rightIndex=None,
        height=0,
        rightlength=0,
        leftlength=0
    ) -> None:
        self.name = rootName if isLeaf is True else None
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
        return self.name

class UPGMA:
    def __init__(
        self, 
        distmat : np.ndarray, 
        linkage_type : str, 
        seq_names : List[str]
    ) -> None:
        assert(distmat.shape[0] == distmat.shape[1])
        assert(len(seq_names) == distmat.shape[0])
        
        self.distmat = distmat
        self.linkage_type = linkage_type
        self.seq_names = seq_names

        self.leafNodeCount = distmat.shape[0]
        self.internalNodeCount = self.leafNodeCount - 1

        self.Tree : List[TreeNode] = []
        for i in range(self.leafNodeCount):
            self.Tree.append(TreeNode(True, self.seq_names[i]))

        self.mapping : np.ndarray = np.arange(self.leafNodeCount)
        self.NN : np.ndarray = np.argmin(self.distmat, dim=1)
        self.minDist : np.ndarray = np.amin(self.distmat, dim=1)

        self.cluter()

    def cluster_one_iteration(self):
        Lmin = np.argmin(self.minDist)
        Rmin = self.NN[Lmin]
        newDist : float 
        newMinDist : float = np.inf
        newNN : int

        for i in range(self.leafNodeCount):
            if i == Lmin or i == Rmin or self.mapping[i] == None:
                continue
            
            if self.linkage_type == 'AVG':
                newDist = (self.distmat[Lmin][i] + self.distmat[Rmin][i]) / 2
            elif self.linkage_type == 'MIN':
                newDist = min(self.distmat[Lmin][i], self.distmat[Rmin][i])
            elif self.linkage_type == 'MAX':
                newDist = max(self.distmat[Lmin][i], self.distmat[Rmin][i])
            else:
                raise NotImplementedError

            self.distmat[Lmin][i] = newDist
            self.distmat[i][Lmin] = newDist

            ## assume C(i,j) = NN(K) <=> C(i) = NN(K) or C(j) = NN(K)
            if self.NN[i] == Rmin or self.NN[i] == Lmin:
                self.NN[i] = Lmin
                self.minDist[i] = newDist

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
        self.Tree.append(TreeNode(False, None, Lindex, Rindex, new_height, Rlength, Llength))

        # Update New Node
        self.mapping[Lmin] = newID
        self.minDist[Lmin] = newMinDist
        self.NN[Lmin] = newNN
        
        # Delete Node
        self.mapping[Rmin] = None
        self.minDist[Rmin] = np.inf
        self.NN[Rmin] = np.inf

    def cluter(self):
        for iter in range(self.internalNodeCount):
            self.cluster_one_iteration()
    
    def writeTree(self, file):
        pass

