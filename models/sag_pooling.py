import dgl
import torch
import torch.nn as nn
import dgl.nn.pytorch as conv
import math

import numpy as np

class SagPooling(nn.Module):
    r"""Apply spp pooling over the nodes in the graph.
    .. math::
        r^{(i)} = \sum_{k=1}^{N_i} x^{(i)}_k
    """
    def __init__(self,input_dim,activation_fn,k):
        super(SagPooling, self).__init__()
        self.layers=nn.ModuleList()
        self.layers.append(conv.GraphConv(input_dim,1,norm=True,bias=True,activation=activation_fn))
        self.k=k
        print('Keeping ',k,' percent nodes')
        
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

            for idx in range(len(self.layers)):
                w = self.layers[idx](graph,features)

            n_graphs=graph.batch_size
            nodes_per_graph=graph.batch_num_nodes
            start=0
            stop=start+nodes_per_graph[0]
            
            for xx in range(n_graphs):

                topK=nodes_per_graph[xx]-math.ceil(self.k*nodes_per_graph[xx])
                out=torch.topk(w[start:stop,:],k=topK,dim=0,largest=False)
                w[out[1]+start,0]=0.0
                
                if xx < n_graphs-1:
                    start=start+nodes_per_graph[xx]
                    stop=start+nodes_per_graph[xx+1]


            # output=features*w

            # np.savetxt('features.txt',features.to("cpu").detach().numpy(),delimiter=' ')
            # np.savetxt('masked_features.txt',output.to("cpu").detach().numpy(),delimiter=' ')
            # np.savetxt('node_info.txt',nodes_per_graph,delimiter=' ')
            # np.savetxt('weights.txt',w.to("cpu").detach().numpy(),delimiter=' ')
            # exit()
            
            return features*w

