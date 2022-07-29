import numpy as np
from typing import Dict, List
from tqdm import tqdm
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
        parentIndex=None, 
        leftIndex=None, 
        rightIndex=None,
        height=0,
        rightlength=0,
        leftlength=0
    ) -> None:
        self.isLeaf = isLeaf
        self.ID     = ID
        self.data   = data if isLeaf is True else None
        self.left   = leftIndex
        self.right  = rightIndex
        self.parent = parentIndex

        self.height = height
        self.rightLength = rightlength
        self.leftLength = leftlength
    
    def getRight(self):
        return self.right
    def getLeft(self):
        return self.left
    def getName(self):
        return self.data['name'] if self.data is not None else f"Internal Node ({self.ID})"

###### BROKEN UPGMA ######
class UPGMA:
    def __init__(
        self, 
        seqs : List[Dict],
        linkage_type : str,
        dist_type : str,
        UPGMA_type : str,
        model = None,
        save_path = None,
    ) -> None:
        
        self.distmat = distMatrix(seqs, dist_type, model, save_path)
        self.linkage_type = linkage_type
        self.UPGMA_type = UPGMA_type
        self.leafNodeCount = self.distmat.shape[0]
        self.internalNodeCount = self.leafNodeCount - 1

        self.Tree : List[TreeNode] = []
        for i in range(self.leafNodeCount):
            self.Tree.append(TreeNode(True, i, seqs[i]))

        if self.UPGMA_type == 'clustal':
            self.UPGMA_clustal()
        elif self.UPGMA_type == 'LCP':
            self.UPGMA_LCP()
        else:
            NotImplementedError
        
        self.rootID = len(self.Tree) - 1

    def UPGMA_clustal(self):
        self.mapping : np.ndarray = np.arange(self.leafNodeCount)
        self.NN : np.ndarray = np.argmin(self.distmat, axis=1)
        self.minDist : np.ndarray = np.amin(self.distmat, axis=1)
        for iter in tqdm(range(self.internalNodeCount), desc="UPGMA"):
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
            self.Tree[Lindex].parent = newID
            self.Tree[Rindex].parent = newID
            self.Tree.append(TreeNode(False, newID, None, None, Lindex, Rindex, new_height, Rlength, Llength))

            # Update New Node
            self.mapping[Lmin] = newID
            self.minDist[Lmin] = newMinDist
            self.NN[Lmin] = newNN
            
            # Delete Node
            self.mapping[Rmin] = -1
            self.minDist[Rmin] = BIG_DIST
            self.NN[Rmin] = -1

    def UPGMA_LCP(self):
        self.matrix2id = np.arange(self.leafNodeCount)
        self.RNN = []
        self.NN = []
        
        for iter in tqdm(range(self.internalNodeCount), desc="UPGMA"):
            if len(self.RNN) == 0:
                for i in range(self.leafNodeCount):
                    if self.matrix2id[i] != -1:
                        cand_MatrixID = i
                        break
                self.RNN.append(cand_MatrixID)
                cand_NN_MatrixID = np.argmin(self.distmat[cand_MatrixID])
                self.NN.append(cand_NN_MatrixID)
            
            # extend RNN
            while len(self.RNN) == 1 or not (self.NN[-2] == self.RNN[-1] and self.NN[-1] == self.RNN[-2]):
                cand_MatrixID = self.NN[-1]
                self.RNN.append(cand_MatrixID)
                cand_NN_MatrixID = np.argmin(self.distmat[cand_MatrixID])
                self.NN.append(cand_NN_MatrixID)
            
            # Reduction
            Lmin = self.RNN[-1]
            Rmin = self.RNN[-2]
            newID = len(self.Tree)
            # Update Tree
            dLR = self.distmat[Lmin][Rmin]
            new_height = dLR / 2
            Lindex = self.matrix2id[Lmin]
            Rindex = self.matrix2id[Rmin]
            Llength = new_height - self.Tree[Lindex].height
            Rlength = new_height - self.Tree[Rindex].height
            self.Tree[Lindex].parent = newID
            self.Tree[Rindex].parent = newID
            self.Tree.append(TreeNode(False, newID, None, None, Lindex, Rindex, new_height, Rlength, Llength))
            # Update matrix2id
            self.matrix2id[Lmin] = newID
            self.matrix2id[Rmin] = -1
            # Update RNN/NN
            self.RNN = self.RNN[:-2]
            self.NN = self.NN[:-2]
            if len(self.RNN) != 0:
                self.NN[-1] = np.argmin(self.distmat[self.RNN[-1]])
            # Update Distance Matrix
            for i in range(self.leafNodeCount):
                if self.matrix2id[i] == -1:
                    continue
                    
                if self.linkage_type == 'AVG':      # UPGMA
                    newDist = (self.distmat[i][Lmin] + self.distmat[i][Rmin]) / 2
                elif self.linkage_type == 'MIN':    # Single-Linkage
                    newDist = min(self.distmat[i][Lmin], self.distmat[i][Rmin])
                else:
                    raise NotImplementedError

                self.distmat[Lmin][i] = self.distmat[i][Lmin] = newDist
                self.distmat[i][Rmin] = BIG_DIST
    
    def writeTree(self, file):
        visited = np.zeros(len(self.Tree))
        # num_vis = 0
        root = self.Tree[self.rootID]
        with open(file, 'w') as f:
            # self.DFS_output(self.Tree[self.rootID], f, -1)
            while True:
                if visited[root.ID] == 0:
                    visited[root.ID] = 1
                    # num_vis += 1
                if root.isLeaf:
                    parent = self.Tree[root.parent]
                    if parent.left == root.ID:
                        f.write(f"{root.getName()}:{parent.leftLength}\n")
                    elif parent.right == root.ID:
                        f.write(f"{root.getName()}:{parent.rightLength}\n")
                    else:
                        print("ERR !!")
                    root = parent
                elif visited[root.left] == 0:
                    f.write('(\n')
                    root = self.Tree[root.left]
                elif visited[root.right] == 0:
                    f.write(',\n')
                    root = self.Tree[root.right]
                else:
                    if root.parent == None: # root
                        f.write(f")\n")
                        # print(num_vis, len(self.Tree))
                        break
                    else:
                        parent = self.Tree[root.parent]
                        if parent.left == root.ID:
                            f.write(f"):{parent.leftLength}\n")
                        elif parent.right == root.ID:
                            f.write(f"):{parent.rightLength}\n")
                        else:
                            print("ERR !!")
                        root = parent
            f.write(';')
    
    def DFS_output(self, root : TreeNode, f, length):
        if root.isLeaf:
            f.write(f"{root.getName()}:{length}\n")
            return
        
        f.write('(\n')
        L, R = root.left, root.right
        self.DFS_output(self.Tree[L], f, root.leftLength)
        f.write(',\n')
        self.DFS_output(self.Tree[R], f, root.rightLength)
        if length != -1: # root
            f.write(f"):{length}\n")
        else:
            f.write(f")\n")
        return 
    
    def appendTree(self, cluster, clusterID):
        node2replace = self.Tree[clusterID]
        assert node2replace.ID == clusterID
        
        parentNode = self.Tree[node2replace.parent]
        # New ID of subtree root 
        updated_rootID = self.DFS_update(cluster.Tree[cluster.rootID], cluster)
        # Link subtree root to the right popsition
        if parentNode.getLeft() == clusterID:
            parentNode.left = updated_rootID
        else:
            parentNode.right = updated_rootID
        
        # Recursively update every new node's parentID 
        self.update_parent_ID(updated_rootID, node2replace.parent)
        
    def DFS_update(self, root : TreeNode, cluster):
        if root.isLeaf:
            newID = len(self.Tree)
            self.Tree.append(TreeNode(True, newID, root.data))
            return newID
        
        
        left_new_ID = self.DFS_update(cluster.Tree[root.getLeft()], cluster)
        right_new_ID = self.DFS_update(cluster.Tree[root.getRight()], cluster)

        newID = len(self.Tree)
        self.Tree.append(TreeNode(
            isLeaf      = False, 
            ID          = newID, 
            data        = None, 
            parentIndex = None,         # will be updated later!!
            leftIndex   = left_new_ID, 
            rightIndex  = right_new_ID, 
            height      = root.height, 
            rightlength = root.rightLength, 
            leftlength  = root.leftLength,
        ))
        return newID
    
    def update_parent_ID(self, rootID : int, parentID):
        self.Tree[rootID].parent = parentID
        if self.Tree[rootID].isLeaf:
            return
        self.update_parent_ID(self.Tree[rootID].getLeft(), rootID)
        self.update_parent_ID(self.Tree[rootID].getRight(), rootID)
    
    def print_topology(self):
        self.DFS_print(self.Tree[self.rootID])
    
    def DFS_print(self, root : TreeNode):
        if root.isLeaf:
            print(f"{root.getName()}")
            return
        
        print('(')
        L, R = root.left, root.right
        self.DFS_print(self.Tree[L])
        print(',')
        self.DFS_print(self.Tree[R])
        print(f") : {root.getName()}")