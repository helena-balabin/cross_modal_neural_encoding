# Structural Analysis

## Dataset

The structural analysis uses the preprocessed COCO-A dataset `helena-balabin/coco_a_preprocessed_all`, which provides graph-based structural annotations for all image–caption pairs in COCO-Actions.

### AMR Graphs (text)

For each caption, an Abstract Meaning Representation (AMR) graph generated with AMRBart (Bai et al., 2022) encodes sentence meaning as a rooted directed acyclic graph. Three scalar properties are extracted:

| Column | Description |
|---|---|
| `amr_n_nodes` | Number of semantic concept nodes |
| `amr_n_edges` | Number of semantic relation edges |
| `amr_graph_depth` | Longest shortest path between any two nodes |

### Action Graphs (images)

For each image, a human-annotated action graph from COCO-Actions constrains scene relations to a fixed vocabulary of visual action verbs. Three parallel properties are extracted:

| Column | Description |
|---|---|
| `coco_a_nodes` | Number of entities (humans, animals, objects) |
| `coco_a_edges` | Number of visual interactions (edges) |
| `coco_a_graph_depth` | Graph depth measuring interaction complexity |

Embeddings are matched to graph type by modality: text embeddings are probed with AMR targets, vision embeddings with action-graph targets.

---

## Structural Probing (§3.2)

Implemented in `cross_modal_neural_encoding/modeling/structural_probing.py`.

```bash
python -m cross_modal_neural_encoding.modeling.structural_probing
```

### Method

A separate ridge regression is fit per (encoder, target) pair on frozen encoder embeddings. The regularisation strength α is selected via **nested 5-fold cross-validation**:

- **Outer loop** (`GroupKFold`, 5 folds): evaluates R² on held-out stimuli; R² is averaged across outer folds.
- **Inner loop** (`GroupKFold`, 5 folds on the outer training portion): selects the best α from the grid `{10⁻³, 10⁻², …, 10³}`.

Probing is applied to the **entire COCO-A dataset** (not just the 252 fMRI stimuli) to obtain reliable R² estimates of each encoder's representational capacity for compositional structure.

### Metric

Performance is the coefficient of determination R² averaged across outer test folds. R² > 0 indicates that the encoder's embedding contains linearly decodable structural information.

### Interpretation

Comparing R² between text and vision encoders reveals whether compositional graph properties are represented equally across modalities. The paper finds that text encoders represent these properties substantially better than vision encoders.


The residual encoding analysis that uses these structural targets is documented separately: see [Residual Neural Encoding](residual_encoding.md).
