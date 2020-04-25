"""Torch modules for graph convolutions."""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from dgl import function as fn
import numpy as np
import dgl.nn.pytorch as conv

# pylint: enable=W0235
class GcapsConv(nn.Module):
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
                 num_gfc_layers=2,
                 num_stats_in=1,
                 num_stats_out=1,
                 activation=None):
        super(GcapsConv, self).__init__()
        self._in_feats        = in_feats
        self._out_feats       = out_feats
        self._num_stats_in    = num_stats_in
        self._num_stats_out   = num_stats_out
        self._num_gfc_layers  = num_gfc_layers
        self._activation_func = activation

        self._gin=conv.GINConv(None,'sum')
        self._stat_layers = nn.ModuleList()
        for _ in range(self._num_stats_out):
            gfc_layers = nn.ModuleList()
            curr_input_dim = self._in_feats * self._num_stats_in
            for _ in range(self._num_gfc_layers):
                gfc_layers.append(nn.Linear(curr_input_dim,self._out_feats))
                curr_input_dim = self._out_feats

            self._stat_layers.append(gfc_layers)


    
    def reset_parameters(self):
         """Reinitialize learnable parameters."""
         gain=nn.init.calculate_gain('relu')

         for i in range(self._num_stats_out):
             for j in range(self._num_gfc_layers):
                 nn.init.xavier_normal(self._stat_layers[i][j].weight,gain=gain)


    def forward(self, graph, x_in):
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

        norm = th.pow(graph.in_degrees().float().clamp(min=1), -0.5)
        shp = norm.shape + (1,) * (x_in.dim() - 1)
        norm = th.reshape(norm, shp).to(x_in.device)
        x_in = x_in*norm
        
        x = x_in

        output = []
        for i in range(self._num_stats_out):
            out = self._gin(graph,x)
            for j in range(self._num_gfc_layers):
                out = self._stat_layers[i][j](out)
                out = self._activation_func(out)

            output.append(out)    
            x = th.mul(x, x_in)

            
        output = th.cat(output,dim=-1)
        return output

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}, p_in={_num_stats_in}, p_out={_num_stats_out}, gfc={_num_gfc_layers}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        if '_reducer' in self.__dict__:
            summary += ', reducer={_reducer}'

        return summary.format(**self.__dict__)

