import dgl
import torch
import torch.nn as nn
import numpy as np


class SppPooling(nn.Module):
    r"""Apply spp pooling over the nodes in the graph.
    .. math::
        r^{(i)} = \sum_{k=1}^{N_i} x^{(i)}_k
    """
    def __init__(self,input_dim,grid):
        super(SppPooling, self).__init__()
        self.input_dim=input_dim
        self.grid=grid

        
    def forward(self, graph, features, xy):
        r"""Compute sum pooling.
        Parameters
        ----------
        graph : DGLGraph or BatchedDGLGraph
            The graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, *)` where
            :math:`N` is the number of nodes in the graph.
        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(*)` (if
            input graph is a BatchedDGLGraph, the result shape
            would be :math:`(B, *)`.
        """
        with graph.local_scope():
            n_graphs=graph.batch_size
            nodes_per_graph=graph.batch_num_nodes
            output=torch.zeros([n_graphs,self.grid,self.grid,self.input_dim]).to(features.device)

            start=0
            stop=start+nodes_per_graph[0]
            
            for xx in range(n_graphs):
                
                for idx in range(start,stop):
                    output[xx,xy[idx,0],xy[idx,1],:]+=(features[idx,:]/np.float64(xy[idx,2]))

                if xx < n_graphs-1:
                    start=start+nodes_per_graph[xx]
                    stop=start+nodes_per_graph[xx+1]
                    
            return output

