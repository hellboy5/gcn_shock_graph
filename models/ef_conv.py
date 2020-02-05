"""Torch modules for graph convolutions."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from dgl import function as fn
import numpy as np

# pylint: enable=W0235
class EfConv(nn.Module):
    r"""Apply graph convolution over an input signal.

    Graph convolution is introduced in `GCN <https://arxiv.org/abs/1609.02907>`__
    and can be described as below:

    .. math::
      h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})

    where :math:`\mathcal{N}(i)` is the neighbor set of node :math:`i`. :math:`c_{ij}` is equal
    to the product of the square root of node degrees:
    :math:`\sqrt{|\mathcal{N}(i)|}\sqrt{|\mathcal{N}(j)|}`. :math:`\sigma` is an activation
    function.

    The model parameters are initialized as in the
    `original implementation <https://github.com/tkipf/gcn/blob/master/gcn/layers.py>`__ where
    the weight :math:`W^{(l)}` is initialized using Glorot uniform initialization
    and the bias is initialized to be zero.

    Notes
    -----
    Zero in degree nodes could lead to invalid normalizer. A common practice
    to avoid this is to add a self-loop for each node in the graph, which
    can be achieved by:

    >>> g = ... # some DGLGraph
    >>> g.add_edges(g.nodes(), g.nodes())


    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    norm : bool, optional
        If True, the normalizer :math:`c_{ij}` is applied. Default: ``True``.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation: callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.

    Attributes
    ----------
    weight : torch.Tensor
        The learnable weight tensor.
    bias : torch.Tensor
        The learnable bias tensor.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 edge_dim,
                 aggregate,
                 bias=True,
                 activation=None):
        super(EfConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self._edge_dim = edge_dim

        if aggregate == 'sum':
            self._reducer=fn.sum
        elif aggregate == 'mean':
            self._reducer=fn.mean
        elif aggregate == 'max':
            self._reducer = fn.max
        else:
            raise KeyError('Aggregator type {} not recognized: '.format(aggregate))

        self.lin = nn.Linear(in_feats,out_feats,bias=bias)
        self._activation = activation
        
        self.reset_parameters()



    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain=nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.lin.weight, gain=gain)

    def forward(self, graph, node_feat,edge_feat):
        r"""Compute graph convolution.

        Notes
        -----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature

        Returns
        -------
        torch.Tensor
            The output feature
        """
        graph = graph.local_var()

        # aggregate first then mult W
        fstack=[]
        for idx in range(self._edge_dim):

            if node_feat.shape[1]==self._in_feats:
                graph.ndata['h']=node_feat
            else:
                start=idx*self._in_feats
                stop=idx*self._in_feats+self._in_feats
                graph.ndata['h']=node_feat[:,start:stop]
                
            graph.edata['ef']=edge_feat[:,idx]
            graph.update_all(fn.u_mul_e('h','ef','m'),
                             self._reducer('m','h'))
            rst = graph.ndata['h']
            rst = self.lin(rst)

            fstack.append(rst)

            graph.edata.pop('ef')


        rst = th.cat(fstack,dim=-1)
        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}, edge={_edge_dim}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        if '_reducer' in self.__dict__:
            summary += ', reducer={_reducer}'

        return summary.format(**self.__dict__)

