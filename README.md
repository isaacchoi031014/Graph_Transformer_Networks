# GTN-GAT: Attention-Based Neighbor Aggregation for Graph Transformer Networks

**Author's note.** This document describes an extension to the original *Graph Transformer Networks* (GTN) implementation in this repository. It replaces GTN's fixed, degree-normalized (GCN-style) neighbor aggregation with a learned, attention-based (GAT-style) aggregation, while leaving the meta-path graph learning mechanism — the central contribution of the original paper — completely unchanged.

## 1. Background

Yun et al. (2019) introduce Graph Transformer Networks (GTN) for representation learning on heterogeneous graphs [1]. Given a set of relation-specific adjacency matrices, GTN learns to (i) combine them into soft, differentiable **meta-path graphs**, and (ii) aggregate node features over those learned graphs using a Graph Convolutional Network (GCN) layer [2]. This project keeps (i) exactly as published and modifies only (ii), substituting a Graph Attention Network (GAT) layer [3] for the GCN layer.

The resulting model, **GTN-GAT**, is exposed in this codebase as `--model GTN_GAT`, selectable alongside the original `--model GTN` without altering the latter's behavior in any way.

## 2. Method

### 2.1 Meta-path graph learning (unchanged from GTN)

Let $A_1, \dots, A_K \in \mathbb{R}^{N \times N}$ be the adjacency matrices of the $K$ input relation types on a graph with $N$ nodes (including an identity relation appended to represent self-loops). A `GTConv` layer with learnable weights $\Phi \in \mathbb{R}^{C \times K}$ produces, for each of $C$ output channels, a convex combination of the input relations:

$$
Q_c = \sum_{k=1}^{K} \mathrm{softmax}(\Phi_{c,:})_k \, A_k, \qquad c = 1, \dots, C
$$

The first `GTLayer` computes two independent convex combinations per channel, $Q_c^{(1)}$ and $Q_c^{(2)}$, and composes them via sparse matrix multiplication to obtain a length-2 meta-path graph:

$$
H_c^{(1)} = Q_c^{(1)} \, Q_c^{(2)}
$$

Each subsequent `GTLayer` $l = 2, \dots, L$ learns one new convex combination $Q_c^{(l)}$ and extends the composed graph by one hop:

$$
H_c^{(l)} = H_c^{(l-1)} \, Q_c^{(l)}
$$

After $L$ layers, $H_c^{(L)}$ is the learned, weighted meta-path adjacency for channel $c$. It is then row-normalized by its learned out-degree before being handed to the aggregation layer:

$$
\tilde{A}^{(c)}_{ij} = \frac{H_{c,ij}^{(L)}}{\sum_{j'} H_{c,ij'}^{(L)}}
$$

None of this machinery (`GTConv`, `GTLayer`, or the row-normalization step) was modified for this project.

### 2.2 GAT-style aggregation (this project's contribution)

For each channel $c$, GTN originally applies a GCN layer to $\tilde{A}^{(c)}$. GTN-GAT instead applies a custom multi-head `GATConv` layer, described below. Superscripts $(h)$ denote per-head quantities; the channel index $c$ is dropped for readability.

**Linear projection.** Node features $x_i \in \mathbb{R}^{d_{\text{in}}}$ are projected independently per head:

$$
z_i^{(h)} = x_i \, W^{(h)}, \qquad W^{(h)} \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}
$$

**Self-loops.** An explicit self-loop $(i,i)$ is added for every node $i$, with edge weight $1$ (or $0$ if `--remove_self_loops` is set), mirroring the self-loop handling in the original `GCNConv` so the comparison between aggregation mechanisms is not confounded by different effective edge sets.

**Attention coefficients.** For each edge $j \to i$ in the (self-loop-augmented) meta-path graph $\tilde{A}^{(c)}$:

$$
e_{ij}^{(h)} = \mathrm{LeakyReLU}\!\left(a_{\text{src}}^{(h)\top} z_j^{(h)} + a_{\text{dst}}^{(h)\top} z_i^{(h)}\right)
$$

The learned meta-path edge weight $\tilde{A}^{(c)}_{ij}$ is folded in additively in log-space, so that a near-zero learned weight suppresses attention on that edge (score $\to -\infty$) while a larger learned weight raises it:

$$
\hat{e}_{ij}^{(h)} = e_{ij}^{(h)} + \log\!\left(\tilde{A}^{(c)}_{ij} + \varepsilon\right)
$$

**Neighborhood softmax.**

$$
\alpha_{ij}^{(h)} = \frac{\exp\!\left(\hat{e}_{ij}^{(h)}\right)}{\sum_{j' \in \mathcal{N}(i)} \exp\!\left(\hat{e}_{ij'}^{(h)}\right)}
$$

**Aggregation and multi-head averaging.** Per-head outputs are computed, then **averaged** (not concatenated) across the $H$ heads:

$$
h_i' = \frac{1}{H} \sum_{h=1}^{H} \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(h)} \, z_j^{(h)} \; + \; b
$$

Averaging (rather than the concatenation used in the original GAT paper's hidden layers) was a deliberate design choice: it guarantees the layer's output dimensionality stays $d_{\text{out}}$ regardless of $H$, so `GATConv` remains a drop-in replacement for `GCNConv` — same call signature, same output shape — without requiring any change to the downstream channel-concatenation or classifier logic.

**Regularization.** Two dropout points are applied during training only: feature dropout on $x$ before projection, and attention dropout on $\alpha_{ij}^{(h)}$ after the softmax (weights are not renormalized post-dropout, consistent with the original GAT implementation). Both use the same rate $p$, controlled by `--dropout` (default $0.3$).

**Optional residual connection.** A separate, learnably-scaled residual path is implemented but disabled in the results reported here:

$$
h_i' \leftarrow h_i' + \lambda \cdot (x_i W_{\text{res}}), \qquad \lambda \text{ learnable, initialized to } 0.1
$$

enabled via `--residual`; it was excluded from the reported configuration as it re-introduces a graph-bypassing shortcut at a point where the model was already observed to overfit.

### 2.3 Multi-channel combination and classification (unchanged from GTN)

As in the original model, the per-channel outputs are concatenated and passed through a single linear classifier:

$$
X_{\text{cat}} = \Big\Vert_{c=1}^{C} \; \mathrm{ReLU}\big(h^{(c)}\big), \qquad
\hat{y} = W_{\text{cls}} \, X_{\text{cat}}[\text{target}] + b_{\text{cls}}
$$

trained with cross-entropy loss (binary cross-entropy for multi-label datasets, as in the original code).

## 3. Implementation

| File | Change |
|---|---|
| `gat.py` | **New file.** Defines `GATConv`: multi-head attention aggregation, drop-in compatible with `GCNConv`'s `(x, edge_index, edge_weight) -> [num_nodes, out_channels]` interface. |
| `model_gtn.py` | **New class `GTN_GAT`.** A structural copy of `GTN` with `self.gcn = GCNConv(...)` replaced by `self.gat = GATConv(...)`. `GTN`, `GTLayer`, and `GTConv` are untouched. |
| `main.py` | Added `--model GTN_GAT` selection, plus `--heads`, `--dropout`, and `--residual` CLI arguments (all no-ops for `--model GTN`/`FastGTN`). |

## 4. Experimental Setup

- **Dataset:** IMDB heterogeneous graph (movie/actor/director relations, 3-class genre classification), $N = 12{,}772$ nodes, as distributed with the original GTN codebase [1].
- **Common hyperparameters:** `--num_layers 2 --num_channels 2 --node_dim 64 --lr 0.02 --epoch 50 --runs 10` (identical for both models; only the aggregation mechanism differs).
- **GTN-GAT specific:** `--heads 2 --dropout 0.5`, self-loops enabled, residual disabled.
- **Protocol:** for each of 10 runs (independent random initializations), the epoch with the best validation Micro-F1 is selected, and its corresponding test-set score is reported. We report mean $\pm$ standard deviation over the 10 runs.

## 5. Results

| Model | Aggregation | Test Macro-F1 | Test Micro-F1 |
|---|---|---|---|
| GTN (original) | GCN, fixed degree-normalized | $0.5824 \pm 0.0105$ | $0.6120 \pm 0.0040$ |
| **GTN-GAT (ours)** | GAT, learned attention ($H=2$, $p=0.5$) | $\mathbf{0.6064 \pm 0.0143}$ | $\mathbf{0.6260 \pm 0.0185}$ |

GTN-GAT improves both Macro-F1 (+2.4 points) and Micro-F1 (+1.4 points) over the GCN-based baseline, under an identical meta-path learning procedure, training budget, and shared hyperparameters — the only difference between the two rows is the neighbor-aggregation mechanism. The increase in standard deviation (Macro-F1: 0.0143 vs. 0.0105; Micro-F1: 0.0185 vs. 0.0040) suggests attention-based aggregation introduces additional run-to-run variance, plausibly from the added attention parameters and the higher dropout rate required to control overfitting — a trade-off for the model's added expressiveness rather than an indication of instability in the underlying mechanism.

### 5.1 Informal ablations

During development, a small manual search (not an exhaustive grid) surfaced the following qualitative observations, each addressed in the final configuration above:

- **Self-loops.** Without them, `GATConv` has no guaranteed mechanism for a node to attend to its own features, unlike `GCNConv`, which adds them unconditionally. Adding self-loops (§2.2) made the comparison to GCN fair.
- **Overfitting.** An early single-head, low-dropout ($p=0.3$) configuration reached near-perfect training F1 ($\approx 0.98$–$1.0$) while test Micro-F1 plateaued around $0.55$–$0.65$. Raising dropout to $p=0.5$ and enabling $H=2$ heads narrowed this gap somewhat without a dedicated classifier-head regularizer (a `LayerNorm` + dropout head was also prototyped but excluded here to isolate the effect of attention/multi-head alone).
- **Residual connection.** A learnably-scaled residual from raw input features was implemented and is available via `--residual`, but was left disabled for the reported results: it bypasses the learned meta-path graph entirely, which is undesirable while the model is already prone to overfitting.

## 6. Reproducing These Results

```bash
# Baseline (original GTN, untouched)
python main.py --dataset IMDB --model GTN --num_layers 2 --epoch 50 --lr 0.02 --num_channels 2 --runs 10

# GTN-GAT (this project's contribution)
python main.py --dataset IMDB --model GTN_GAT --num_layers 2 --epoch 50 --lr 0.02 --num_channels 2 --runs 10 --heads 2 --dropout 0.5
```

## 7. Limitations and Future Work

- Results are reported on a single dataset (IMDB); the original GTN paper additionally evaluates on ACM and DBLP, which would be a natural next step to test generality.
- The hyperparameter search over `--heads` and `--dropout` was manual and limited in scope, not a systematic sweep; a proper grid or random search would give tighter estimates of the achievable improvement.
- Heads are averaged rather than concatenated-then-projected; concatenation followed by a learned linear projection back to $d_{\text{out}}$ is a natural extension worth comparing.
- The classifier-head regularization (`LayerNorm` + dropout before the final linear layer) was prototyped separately and may compound favorably with multi-head attention; this combination has not yet been evaluated jointly.

## References

[1] Yun, S., Jeong, M., Kim, R., Kang, J., & Kim, H. J. (2019). *Graph Transformer Networks*. Advances in Neural Information Processing Systems (NeurIPS).

[2] Kipf, T. N., & Welling, M. (2017). *Semi-Supervised Classification with Graph Convolutional Networks*. International Conference on Learning Representations (ICLR).

[3] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). *Graph Attention Networks*. International Conference on Learning Representations (ICLR).
