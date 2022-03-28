import torch
import copy
import numpy as np
from typing import Dict, List
from src.kmeans_pytorch import kmeans

__all__ = [
    'BisectingKmeans',
]

def BisectingKmeans(seqs : List[Dict], device, min_cluster_size=100):
    final_cluster = [{
        'center' : None,
        'seqs' : copy.deepcopy(seqs)
    }]
    
    while len(final_cluster[0]['seqs']) > min_cluster_size:
        # Extract MAX
        biggest_cluster = final_cluster[0]
        clusterPoints = biggest_cluster['seqs']
        # POP
        final_cluster = final_cluster[1:]
        
        # make coordinate array
        x = []
        for seq in clusterPoints:
            x.append(seq['embedding'])
        x = torch.from_numpy(np.array(x))
        cluster_ids, cluster_centers = kmeans(
            X = x,
            num_clusters = 2,
            distance = 'euclidean',
            device = device,
        )
        
        newCluster = [[], []]
        assert len(cluster_ids) == len(clusterPoints)
        
        for clusterID, seq in zip(cluster_ids, clusterPoints):
            newCluster[clusterID].append(seq)
        
        for center, seq in zip(cluster_centers, newCluster):
            if len(seq) == 0:
                continue
            final_cluster.append({
                'center' : center,
                'seqs' : seq
            })
        final_cluster = sorted(final_cluster, key=lambda x : len(x['seqs']), reverse=True)
        
    centers, clusters = [], []
    for clusterID, ele in enumerate(final_cluster):
        center = ele['center']
        seqs = ele['seqs']
        
        assert len(seqs) != 0
        
        centers.append({
            'name' : f"precluster-{clusterID}",
            'embedding' : center
        })
        for seq in seqs:
            seq['cluster'] = clusterID
        clusters.append(seqs)
    return centers, clusters

    