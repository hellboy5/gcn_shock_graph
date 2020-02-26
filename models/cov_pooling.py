import dgl
import torch
import torch.nn as nn
import numpy as np


class CovPooling(nn.Module):
    r"""Apply spp pooling over the nodes in the graph.
    .. math::
        r^{(i)} = \sum_{k=1}^{N_i} x^{(i)}_k
    """
    def __init__(self,hidden_dim):
        super(CovPooling, self).__init__()
        self.hidden_dim=hidden_dim
        
    def forward(self, graph, features):
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
            output=torch.zeros([n_graphs,int(((self.hidden_dim+1)*self.hidden_dim)/2)]).to(features.device)
            start=0
            stop=start+nodes_per_graph[0]

            for xx in range(n_graphs):

                F=features[start:stop,:]

                mu=torch.mean(F,0)
                scale_matrix=F-mu
                cov_matrix=torch.mm(scale_matrix.T,scale_matrix)/F.shape[0]

                idx=torch.triu_indices(cov_matrix.shape[0],cov_matrix.shape[1])
                diag_values=cov_matrix[idx[0,:],idx[1,:]]
                output[xx,:]=diag_values

                if xx < n_graphs-1:
                    start=start+nodes_per_graph[xx]
                    stop=start+nodes_per_graph[xx+1]

            return output

