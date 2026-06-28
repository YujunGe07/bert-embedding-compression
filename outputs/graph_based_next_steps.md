# Graph-Based Compression Next Steps

Antonio's latest guidance suggests keeping the project focused on interpretable PCA-related methods rather than adding learned black-box baselines for now.

## Current Direction

The next experiment compares:

- standard PCA, using covariance eigenvectors;
- sparse precision eigenbasis, using graphical Lasso to estimate a sparse inverse covariance matrix;
- random projection, as a fast non-adaptive baseline.

The sparse precision matrix can be interpreted as a graph over embedding dimensions. Nonzero entries indicate conditional dependencies between dimensions, which may reveal structure that is not visible from the covariance matrix alone.

## First Implementation

Notebook:

`outputs/bert_compression_research_graph_based.ipynb`

New section:

`Step 12: Graph-Based / Sparse Precision Compression Basis`

This section reuses the rigorous PCA/classifier split:

- fit transform on the PCA-fitting split;
- train classifier on the classifier-training split;
- evaluate on the held-out AG News test set.

## What to Look For

The sparse precision eigenbasis is promising if it:

- performs near standard PCA at the same compressed dimension;
- clearly beats random projection;
- has meaningful graph sparsity;
- transfers better than PCA under held-out category or cross-task tests.

## Follow-Up Experiments

- Run the sparse precision basis under the held-out Sci/Tech stress test.
- Fit transform on AG News and evaluate on SST-2 or another NLP task.
- Compare subspace similarity between PCA bases and graph-based bases across datasets.
- Measure fit/runtime cost versus accuracy.
- Explore STAC-USC graph learning methods if the simple graphical Lasso baseline looks promising.
