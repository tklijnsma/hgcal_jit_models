import os
import os.path as osp
import math

import numpy as np
import torch
import gc
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch.utils.checkpoint import checkpoint
from torch_cluster import knn_graph

from torch_geometric.nn import EdgeConv, NNConv
from torch_geometric.nn.pool.edge_pool import EdgePooling

from torch_geometric.utils import normalized_cut
from torch_geometric.utils import remove_self_loops
# from torch_geometric.utils.undirected import to_undirected
from torch_geometric.nn import (
    graclus,
    # max_pool,
    # max_pool_x,
    global_mean_pool, global_max_pool,
    global_add_pool
    )

transform = T.Cartesian(cat=False)

def normalized_cut_2d(edge_index, pos):
    # TK: Avoid expansion, just refer by index
    # row, col = edge_index
    row = edge_index[0]
    col = edge_index[1]
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))

# TK: This is the non-jittable version
class DynamicReductionNetwork(nn.Module):
    # This model iteratively contracts nearest neighbour graphs 
    # until there is one output node.
    # The latent space trained to group useful features at each level
    # of aggregration.
    # This allows single quantities to be regressed from complex point counts
    # in a location and orientation invariant way.
    # One encoding layer is used to abstract away the input features.
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1, k=16, aggr='add',
                 norm=torch.tensor([1./500., 1./500., 1./54., 1/25., 1./1000.])):
        super(DynamicReductionNetwork, self).__init__()

        self.datanorm = nn.Parameter(norm)
        
        self.k = k
        start_width = 2 * hidden_dim
        middle_width = 3 * hidden_dim // 2

        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim//2),            
            nn.ELU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )        
        convnn1 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.ELU(),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.ELU()
                                )
        convnn2 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.ELU(),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.ELU()
                                )                
                
        self.edgeconv1 = EdgeConv(nn=convnn1, aggr=aggr)
        self.edgeconv2 = EdgeConv(nn=convnn2, aggr=aggr)
        
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.ELU(),
                                    nn.Linear(hidden_dim, hidden_dim//2),
                                    nn.ELU(),                                    
                                    nn.Linear(hidden_dim//2, output_dim))
        
        
    def forward(self, data):        
        data.x = self.datanorm * data.x
        data.x = self.inputnet(data.x)
        
        data.edge_index = to_undirected(knn_graph(data.x, self.k, data.batch, loop=False, flow=self.edgeconv1.flow))
        data.x = self.edgeconv1(data.x, data.edge_index)
        
        weight = normalized_cut_2d(data.edge_index, data.x)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data)
        
        data.edge_index = to_undirected(knn_graph(data.x, self.k, data.batch, loop=False, flow=self.edgeconv2.flow))
        data.x = self.edgeconv2(data.x, data.edge_index)
        
        weight = normalized_cut_2d(data.edge_index, data.x)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_max_pool(x, batch)
        
        return self.output(x).squeeze(-1)

# _______________________________________________________________________________
# Make it jittable

from typing import Optional, Tuple
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse.storage import SparseStorage

# from torch_sparse import coalesce
# TK: Now takes `value` as an optional argument with a default of None.
#     Also moved to end of argument list
#     This change is really specific to this example - need a more general solution.
@torch.jit.script
def coalesce(index: torch.Tensor, m: int, n: int, value: Optional[torch.Tensor] = None, op: str="add"):
    """Row-wise sorts :obj:`value` and removes duplicate entries. Duplicate
    entries are removed by scattering them together. For scattering, any
    operation of `"torch_scatter"<https://github.com/rusty1s/pytorch_scatter>`_
    can be used.

    Args:
        index (:class:`LongTensor`): The index tensor of sparse matrix.
        value (:class:`Tensor`): The value tensor of sparse matrix.
        m (int): The first dimension of corresponding dense matrix.
        n (int): The second dimension of corresponding dense matrix.
        op (string, optional): The scatter operation to use. (default:
            :obj:`"add"`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    sparse_sizes = (m, n)

    storage = SparseStorage(
        row=index[0], col=index[1], value=value,
        sparse_sizes=sparse_sizes,
        is_sorted=False,
        rowptr=None, rowcount=None, colptr=None, colcount=None, csr2csc=None, csc2csr=None
        )
    storage = storage.coalesce(reduce=op)
    return torch.stack([storage.row(), storage.col()], dim=0), storage.value()

# from torch_geometric.utils.undirected import to_undirected
# TK: Changed expansion into get via index
#     num_nodes now explicitely defined as Optional
#     Also uses the modifief `coalesce` function above to accept `value = None`
@torch.jit.script
def to_undirected(edge_index: torch.Tensor, num_nodes: Optional[int] = None):
    r"""Converts the graph given by :attr:`edge_index` to an undirected graph,
    so that :math:`(j,i) \in \mathcal{E}` for every edge :math:`(i,j) \in
    \mathcal{E}`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: :class:`LongTensor`
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    # row, col = edge_index
    row = edge_index[0]
    col = edge_index[1]
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = coalesce(edge_index, value=None, m=num_nodes, n=num_nodes)
    # edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)
    return edge_index

from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import (
    # pool_edge, pool_batch,
    pool_pos
    )

from torch_scatter import scatter

# TK: Unchanged but not easily importable by itself, so just copied here for convenience
def _max_pool_x(cluster, x, size: Optional[int] = None):
    return scatter(x, cluster, dim=0, dim_size=size, reduce='max')

# TK: Unchanged but uses the coalesce defined above now
def pool_edge(cluster, edge_index,
              edge_attr: Optional[torch.Tensor] = None):
    num_nodes = cluster.size(0)
    edge_index = cluster[edge_index.view(-1)].view(2, -1)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    if edge_index.numel() > 0:
        edge_index, edge_attr = coalesce(edge_index, value=edge_attr, m=num_nodes,
                                         n=num_nodes)
    return edge_index, edge_attr

@torch.jit.script
# TK: Made batch optional argument
#     ALSO KILLED A WHOLE IF STATEMENT
def max_pool_x(cluster, x, batch: Optional[torch.Tensor] = None, size: Optional[int] = None):
    r"""Max-Pools node features according to the clustering defined in
    :attr:`cluster`.
    Args:
        cluster (LongTensor): Cluster vector :math:`\mathbf{c} \in \{ 0,
            \ldots, N - 1 \}^N`, which assigns each node to a specific cluster.
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{(N_1 + \ldots + N_B) \times F}`.
        batch (LongTensor): Batch vector :math:`\mathbf{b} \in {\{ 0, \ldots,
            B-1\}}^N`, which assigns each node to a specific example.
        size (int, optional): The maximum number of clusters in a single
            example. This property is useful to obtain a batch-wise dense
            representation, *e.g.* for applying FC layers, but should only be
            used if the size of the maximum number of clusters per example is
            known in advance. (default: :obj:`None`)
    :rtype: (:class:`Tensor`, :class:`LongTensor`) if :attr:`size` is
        :obj:`None`, else :class:`Tensor`
    """
    # TK: Throws error when trying to compile, if batch is None batch.max() is not defined
    # if size is not None:
    #     batch_size = int(batch.max().item()) + 1
    #     return _max_pool_x(cluster, x, batch_size * size), None

    cluster, perm = consecutive_cluster(cluster)
    x = _max_pool_x(cluster, x)
    # TK: Put the following under an if-statement
    # batch = pool_batch(perm, batch)
    if not batch is None:
        batch = batch[perm]
    return x, batch

# TK: Reimplementation of max_pool that takes x, edge_index and batch as arguments rather than `data`
#     x and edge_index are not optional
@torch.jit.script
def max_pool(cluster, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor]=None, transform: bool=None):
    r"""Pools and coarsens a graph given by the
    :class:`torch_geometric.data.Data` object according to the clustering
    defined in :attr:`cluster`.
    All nodes within the same cluster will be represented as one node.
    Final node features are defined by the *maximum* features of all nodes
    within the same cluster, node positions are averaged and edge indices are
    defined to be the union of the edge indices of all nodes within the same
    cluster.
    Args:
        cluster (LongTensor): Cluster vector :math:`\mathbf{c} \in \{ 0,
            \ldots, N - 1 \}^N`, which assigns each node to a specific cluster.
        data (Data): Graph data object.
        transform (callable, optional): A function/transform that takes in the
            coarsened and pooled :obj:`torch_geometric.data.Data` object and
            returns a transformed version. (default: :obj:`None`)
    :rtype: :class:`torch_geometric.data.Data`
    """
    cluster, perm = consecutive_cluster(cluster)

    x = _max_pool_x(cluster, x)
    # x = None if data.x is None else torch_geometric.nn.pool.max_pool._max_pool_x(cluster, data.x)
    edge_index, edge_attr = pool_edge(cluster, edge_index, edge_attr=None)


    if not(batch is None):
        # batch = pool_batch(perm, batch)
        batch = batch[perm]

    return x, edge_index, batch, edge_attr

    # # batch = None if data.batch is None else pool_batch(perm, data.batch)
    # # pos = None if data.pos is None else pool_pos(cluster, data.pos)
    # pos = None

    # data = Batch(batch=batch, x=x, edge_index=index, edge_attr=attr, pos=pos)

    # if transform is not None:
    #     data = transform(data)

    # return data

# TK: Jittable version of the model defined above
class DynamicReductionNetworkJittableLindsey(nn.Module):
    # This model iteratively contracts nearest neighbour graphs 
    # until there is one output node.
    # The latent space trained to group useful features at each level
    # of aggregration.
    # This allows single quantities to be regressed from complex point counts
    # in a location and orientation invariant way.
    # One encoding layer is used to abstract away the input features.
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1, k=16, aggr='add',
                 norm=torch.tensor([1./500., 1./500., 1./54., 1/25., 1./1000.])):
        super(DynamicReductionNetworkJittableLindsey, self).__init__()

        self.datanorm = nn.Parameter(norm)
        
        self.k = k
        start_width = 2 * hidden_dim
        middle_width = 3 * hidden_dim // 2

        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim//2),            
            nn.ELU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )        
        convnn1 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.ELU(),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.ELU()
                                )
        convnn2 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.ELU(),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.ELU()
                                )                
                
        self.edgeconv1 = EdgeConv(nn=convnn1, aggr=aggr).jittable()
        self.edgeconv2 = EdgeConv(nn=convnn2, aggr=aggr).jittable()
        
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.ELU(),
                                    nn.Linear(hidden_dim, hidden_dim//2),
                                    nn.ELU(),                                    
                                    nn.Linear(hidden_dim//2, output_dim))
        
        
    def forward(self, x, batch: Optional[torch.Tensor] = None):
        x = self.datanorm * x
        x = self.inputnet(x)
        
        knn = knn_graph(x, self.k, batch, loop=False, flow=self.edgeconv1.flow)
        edge_index = to_undirected(knn)
        x = self.edgeconv1(x, edge_index)
        
        weight = normalized_cut_2d(edge_index, x)
        cluster = graclus(edge_index, weight, x.size(0))
        edge_attr = None
        # data = max_pool(cluster, data)
        x, edge_index, batch, edge_attr = max_pool(cluster, x, edge_index, batch)
        
        edge_index = to_undirected(knn_graph(x, self.k, batch, loop=False, flow=self.edgeconv2.flow))
        x = self.edgeconv2(x, edge_index)
        
        weight = normalized_cut_2d(edge_index, x)
        cluster = graclus(edge_index, weight, x.size(0))
        x, batch = max_pool_x(cluster, x, batch)

        if not batch is None:
            x = global_max_pool(x, batch)
        
        return self.output(x).squeeze(-1)



class DynamicReductionNetworkJittable(nn.Module):
    # SHAMIK'S VERSION: SAME AS ABOVE WITH SLIGHTLY MORE LAYERS
    def __init__(self, input_dim=5, hidden_dim=64, output_dim=1, k=16, aggr='add',
                 norm=torch.tensor([1./500., 1./500., 1./54., 1/25., 1./1000.])):
        super(DynamicReductionNetworkJittable, self).__init__()

        self.datanorm = nn.Parameter(norm)
        
        self.k = k
        start_width = 2 * hidden_dim
        middle_width = 3 * hidden_dim // 2

        
        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim*2),            
            nn.ELU(),
            nn.Linear(hidden_dim*2, hidden_dim*2),
            nn.ELU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ELU(),
        )
                
        convnn1 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.ELU(),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.ELU()
                                )
        convnn2 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.ELU(),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.ELU()
                                )
        
        convnn3 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.ELU(),
                                nn.Linear(middle_width, hidden_dim),                                             
                                nn.ELU()
                                )
                
        self.edgeconv1 = EdgeConv(nn=convnn1, aggr=aggr).jittable()
        self.edgeconv2 = EdgeConv(nn=convnn2, aggr=aggr).jittable()
        self.edgeconv3 = EdgeConv(nn=convnn3, aggr=aggr).jittable()
        
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.ELU(),
                                    nn.Linear(hidden_dim, hidden_dim//2),
                                    nn.ELU(),
                                    nn.Linear(hidden_dim//2, output_dim)
                                   )


    def forward(self, x, batch: Optional[torch.Tensor] = None):
        x = self.datanorm * x
        x = self.inputnet(x)
        
        edge_index = to_undirected(knn_graph(x, self.k, batch, loop=False, flow=self.edgeconv1.flow))
        x = self.edgeconv1(x, edge_index)        
        weight = normalized_cut_2d(edge_index, x)
        cluster = graclus(edge_index, weight, x.size(0))
        edge_attr = None
        x, edge_index, batch, edge_attr = max_pool(cluster, x, edge_index, batch)

        # Additional layer by Shamik
        edge_index = to_undirected(knn_graph(x, self.k, batch, loop=False, flow=self.edgeconv3.flow))
        x = self.edgeconv1(x, edge_index)        
        weight = normalized_cut_2d(edge_index, x)
        cluster = graclus(edge_index, weight, x.size(0))
        edge_attr = None
        x, edge_index, batch, edge_attr = max_pool(cluster, x, edge_index, batch)
        
        edge_index = to_undirected(knn_graph(x, self.k, batch, loop=False, flow=self.edgeconv2.flow))
        x = self.edgeconv2(x, edge_index)
        
        weight = normalized_cut_2d(edge_index, x)
        cluster = graclus(edge_index, weight, x.size(0))
        x, batch = max_pool_x(cluster, x, batch)

        if not batch is None:
            x = global_max_pool(x, batch)
        
        return self.output(x).squeeze(-1)


nonjit_model = DynamicReductionNetworkJittable(input_dim=5, hidden_dim=64, output_dim=1, k=16)
model = torch.jit.script(nonjit_model)
print(model)
torch.jit.save(model, 'drn_noweights.pt')
print('okay')

