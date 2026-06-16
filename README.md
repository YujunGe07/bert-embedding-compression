# BERT Embedding Compression

Preliminary research project on compressing BERT text embeddings with PCA while preserving downstream classification performance.

This repository is a work in progress. The current experiments use AG News classification with `google-bert/bert-base-uncased` `[CLS]` embeddings, PCA compression, and logistic regression evaluation.

## Current Contents

- `outputs/bert_compression_research_after_meeting.ipynb`  
  Executed notebook with preliminary experiments and results.

- `outputs/bert_compression_summary.tex`  
  Short LaTeX summary prepared for advisor feedback.

## Current Results

Initial AG News experiment:

- 768-dimensional BERT baseline accuracy: `0.830`
- 128-dimensional PCA accuracy: `0.834`
- Dimensionality reduction at 128 dimensions: `83.3%`

Stricter split where PCA fitting examples are separated from classifier-training examples:

- 768-dimensional baseline accuracy: `0.822`
- 384-dimensional PCA accuracy: `0.818`
- 512-dimensional PCA accuracy: `0.826`

The notebook also includes an initial held-out category stress test where PCA is fit without seeing the `Sci/Tech` category and then evaluated on the full AG News test set.

## Research Direction

Prior work has already studied dimensionality reduction and compression for pretrained embeddings, so the likely path to novelty is not simply showing that PCA can compress BERT embeddings.

The current direction is to study whether a compression basis learned from one distribution of embeddings can transfer across:

- held-out categories,
- shifted datasets,
- different NLP tasks,
- different embedding models,
- alternative compression methods.

## Possible Next Steps

- Run cross-task transfer experiments, such as AG News to SST-2.
- Compare transferred PCA against in-task PCA.
- Add baselines such as random projection, autoencoder compression, quantization, or graph-based PCA.
- Repeat across multiple random seeds.
- Measure whether PCA subspace similarity or variance explained predicts transfer performance.

## Status

Incomplete preliminary project. Not yet publication-ready.
