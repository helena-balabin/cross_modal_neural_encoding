# Structural Analysis

## Dataset

The structural analysis is based on the preprocessed COCO-A dataset available at `helena-balabin/coco_a_preprocessed_all`. This dataset provides graph-based structural annotations for the text-image pairs used in the COCO-A dataset.

### AMR Graphs for Text

For each caption, an Abstract Meaning Representation (AMR) graph is provided. AMR is a semantic representation that abstracts away from surface-level syntax and focuses on the underlying meaning of a sentence. The AMR graphs are generated using AMRBart.

The following structural properties are extracted from each AMR graph:

- **`amr_n_nodes`**: Number of semantic concepts (nodes) in the graph
- **`amr_n_edges`**: Number of semantic relations (edges) between concepts
- **`amr_graph_depth`**: Graph depth, defined as the longest shortest path between any two nodes, measuring the compositional structure

### Action Graphs for Images

For each image, an action graph is provided based on the COCO-Actions dataset. These graphs capture human-object and human-human interactions in the scene using a fixed vocabulary of visual action verbs.

The following structural properties are extracted from each action graph:

- **`coco_a_nodes`**: Number of entities (humans, animals, objects) in the graph
- **`coco_a_edges`**: Number of interactions (edges) between entities
- **`coco_a_graph_depth`**: Graph depth, measuring the complexity of interactions in the scene

## Structural Probing Framework

The structural probing framework follows the approach of Balabin et al. (2026). For each modality, we probe the VLM encoder's embeddings to see how well they predict the structural properties of the corresponding graphs.

### Implementation

1. **Data Loading**: The structural targets are loaded directly from the Hugging Face dataset `helena-balabin/coco_a_preprocessed_all` using the `datasets` library.

2. **Target Alignment**: The structural targets are aligned with the VLM embeddings by matching COCO IDs. Only stimuli present in both the embedding dataset and the structural dataset are used.

3. **Model Architecture**: For each structural property, we fit a ridge regression model to predict the property value from the VLM embedding. The model uses nested cross-validation:
   - **Inner CV**: 5-fold GroupKFold to select the optimal regularization strength (alpha) from a logarithmic grid spanning $10^{-3}$ to $10^{3}$.
   - **Outer Evaluation**: The final model is evaluated on the complete dataset using the mean optimal alpha from the inner CV.

4. **Evaluation Metric**: Model performance is reported as the coefficient of determination ($R^2$), which measures the proportion of variance in the structural property that is predictable from the embedding.

### Interpretation

High $R^2$ values indicate that the VLM encoder's representation contains substantial information about the structural properties of the input. Comparing $R^2$ values between text and vision encoders can reveal differences in how compositional structure is represented across modalities.

## Residual Encoding Approach

The residual encoding approach is based on Oota et al. (2023) and tests whether the structural properties identified in the probing analysis are what enable cross-modal alignment.

### Method

1. **Structural Feature Vector**: For each stimulus, we construct a 3-dimensional feature vector $\mathbf{s \in \mathbb{R^3$ containing the three structural properties: node count, edge count, and graph depth.

2. **Residualization**: We fit a ridge regression model to predict the VLM embedding $\mathbf{e}$ from the structural feature vector $\mathbf{s}$ on the training data:
   \begin{equation}
   \hat{\mathbf{e = \mathbf{W^{* \mathbf{s}
   \end{equation}
   where $\mathbf{W^{*}$ is the optimal weight matrix. We then subtract the prediction from the original embedding to obtain the structure-residualized embedding:
   \begin{equation}
   \tilde{\mathbf{e = \mathbf{e - \hat{\mathbf{e}
   \end{equation}

3. **Control Condition**: To ensure that any effects are due to the removal of structural information rather than general variance reduction, we run a control condition where the structural feature vector $\mathbf{s}$ is randomly permuted across stimuli before fitting the regression model.

4. **Encoding Analysis**: We re-run the full cross-modal encoding pipeline using the structure-residualized embeddings $\tilde{\mathbf{e}$ instead of the original embeddings $\mathbf{e}$.

### Hypotheses

- If compositional structure is the key to cross-modal alignment, we expect a selective reduction in cross-modal encoding accuracy (image $\to$ text and text $\to$ image) while within-modality accuracy (image $\to$ image and text $\to$ text) remains relatively unchanged.
- The control condition with permuted features should show minimal changes in encoding accuracy, confirming that the effects are specific to the removal of genuine structural information.
