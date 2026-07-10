import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_scatter.composite import scatter_softmax
from torch_geometric.utils import add_self_loops
from inits import glorot, zeros


class GATConv(nn.Module):
    r"""A minimal multi-head Graph Attention (GAT) layer.

    This is a drop-in replacement for GCNConv (see gcn.py) inside GTN.
    It is called exactly the same way:

        out = self.gat(X, edge_index=edge_index.detach(), edge_weight=edge_weight)

    where:
        X           : node feature matrix, shape [num_nodes, in_channels]
        edge_index  : the learned meta-path graph from GTLayer/GTConv,
                       shape [2, num_edges]
        edge_weight : the learned edge weights for that meta-path graph,
                       shape [num_edges]

    and it returns node embeddings of shape [num_nodes, out_channels],
    exactly like GCNConv, so the rest of the GTN model (concatenation
    across channels + final linear classifier) does not need to change.

    Instead of GCN's fixed degree-normalized neighbor average, this layer
    LEARNS an attention weight (alpha_ij) for every edge j -> i, so the
    model can decide which neighbors matter more.

    NOTE on multi-head attention: with heads > 1, each head has its own
    independent weight/attention parameters and computes its own
    [num_nodes, out_channels] embedding; the heads are then AVERAGED
    (not concatenated) so the final output shape stays [num_nodes,
    out_channels] no matter how many heads are used. This is what keeps
    the call signature and the rest of GTN_GAT (channel concatenation +
    classifier) unchanged when heads > 1.

    NOTE on edge direction (must match GCNConv's convention used
    elsewhere in this codebase, since edge_index comes from the same
    GTLayer/GTConv code and must be read the same way):
        edge_index[0] = target node i  (the node receiving the message)
        edge_index[1] = source node j  (the neighbor sending the message)

    NOTE on self-loops: just like GCNConv.norm(), this layer adds an
    explicit self-loop (i, i) for every node before computing attention,
    so every node can always attend to itself. This keeps the comparison
    with GCNConv fair (same effective edge set) and matches GCNConv's use
    of args.remove_self_loops: self-loop weight is 1.0 normally, or 0.0
    when args.remove_self_loops is set.

    NOTE on dropout: two dropout points are used (both only active while
    training, i.e. model.train()):
      1. feature dropout on the input x, before z = X W
      2. attention dropout on alpha, after the neighbor softmax
    """

    def __init__(self, in_channels, out_channels, negative_slope=0.2, eps=1e-8, dropout=0.3, heads=1, args=None):
        super(GATConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        # small constant so log(edge_weight + eps) never sees log(0)
        self.eps = eps
        # dropout probability, used in two places (see forward()):
        #   1. feature dropout: randomly zero some input feature dims
        #   2. attention dropout: randomly zero some attention weights
        # only active during training (self.training, set by model.train()/.eval())
        self.dropout = dropout
        # number of independent attention heads; heads=1 behaves the same
        # as the original single-head layer
        self.heads = heads
        self.args = args

        # 1. Shared linear transform per head: z = X W_h  (same role as
        #    GCNConv.weight, but one independent W per head). Stored as a
        #    single [in_channels, heads * out_channels] matrix and reshaped
        #    to [num_nodes, heads, out_channels] after multiplying, which
        #    is equivalent to (and faster than) a python loop over heads.
        self.weight = nn.Parameter(torch.Tensor(in_channels, heads * out_channels))

        # 2. Attention parameters, one "source" and "destination" vector
        #    PER HEAD, so that, for edge j -> i, head h:
        #       e_ij^h = LeakyReLU(a_src^h . z_j^h + a_dst^h . z_i^h)
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        # bias is shared across heads and added once, after the heads are
        # averaged together (see forward())
        self.bias = nn.Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)

        # ---- 0. add explicit self-loops, same as GCNConv.norm() ----
        # (fair comparison: both layers see the same effective edge set,
        # and every node can attend to / aggregate its own features)
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype, device=x.device)
        edge_weight = edge_weight.view(-1)
        assert edge_weight.size(0) == edge_index.size(1)

        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        loop_weight = torch.full((num_nodes, ),
                                  1 if not self.args.remove_self_loops else 0,
                                  dtype=edge_weight.dtype,
                                  device=edge_weight.device)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)
        # edge_weight now has length == edge_index.size(1) (original edges + self-loops)

        # ---- feature dropout: randomly zero some input feature dimensions ----
        # e.g. a node's feature vector [0.9, 0.4, 0.7, 0.2] might become
        # [0.9, 0.0, 0.7, 0.0] for this forward pass. This stops the model
        # from leaning too hard on any single feature dimension. Matches
        # the original GAT paper (dropout applied to the layer's input).
        x = F.dropout(x, p=self.dropout, training=self.training)

        # ---- 1. transform node features: z = X W, one z per head ----
        # z: [num_nodes, heads, out_channels]
        z = torch.matmul(x, self.weight).view(num_nodes, self.heads, self.out_channels)

        target_i = edge_index[0]   # node receiving the message
        source_j = edge_index[1]   # neighbor sending the message

        z_i = z[target_i]          # [num_edges, heads, out_channels]
        z_j = z[source_j]          # [num_edges, heads, out_channels]

        # ---- 2. raw attention score per edge, per head: ----
        # e_ij^h = LeakyReLU(a_src^h.z_j^h + a_dst^h.z_i^h)
        # e: [num_edges, heads]
        e = (z_j * self.att_src).sum(dim=-1) + (z_i * self.att_dst).sum(dim=-1)
        e = F.leaky_relu(e, negative_slope=self.negative_slope)

        # ---- 3. fold in the GTN-learned edge weight (same for every head) ----
        # Adding log(edge_weight + eps) means: a learned weight of ~0 pushes
        # the score to -inf (that neighbor gets ~0 attention after softmax),
        # while a larger learned weight raises the score, so the GTN's
        # learned meta-path graph still influences the attention.
        if edge_weight is not None:
            e = e + torch.log(edge_weight + self.eps).unsqueeze(-1)

        # ---- 4. softmax over the neighbors of each target node i, per head ----
        # alpha: [num_edges, heads]
        alpha = scatter_softmax(e, target_i, dim=0)

        # ---- attention dropout: randomly zero some attention weights ----
        # Example: node i's neighbors originally get
        #     alpha = {B: 0.60, C: 0.30, D: 0.10}
        # after dropout, e.g. C gets randomly dropped:
        #     alpha = {B: 0.60, C: 0.00, D: 0.10}
        # Note alpha no longer sums to 1 after this - that's expected and
        # matches the original GAT paper (dropped weights are not
        # renormalized). This is like randomly removing some edges during
        # training so the model can't over-rely on one neighbor.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        # ---- 5. aggregate per head: h_i'^h = sum_j alpha_ij^h * z_j^h ----
        # out: [num_nodes, heads, out_channels]
        out = scatter_add(alpha.unsqueeze(-1) * z_j, target_i, dim=0, dim_size=num_nodes)

        # ---- 6. average across heads (NOT concatenate), so the output ----
        # ---- shape stays [num_nodes, out_channels] regardless of heads ----
        out = out.mean(dim=1)

        # ---- 7. bias ----
        out = out + self.bias
        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                    self.out_channels)
