# BERT Embedding Compression

<div align="left">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21F?style=flat&logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Research-blue?style=flat)

</div>

> **Can a compression basis learned on one data distribution transfer to another?**
> This project investigates PCA-based dimensionality reduction of BERT embeddings, with a focus on *cross-distribution transfer*: whether subspaces learned on held-out categories, shifted datasets, and different NLP tasks remain meaningful.

---

## Results

### AG News Classification

BERT `[CLS]` embeddings are compressed and evaluated using logistic regression.

| Setting | Dimensions | Accuracy | Reduction |
|---------|-----------:|---------:|----------:|
| Baseline | 768 | 83.0% | — |
| PCA, joint split | 128 | **83.4%** | **83.3%** |
| PCA, strict split | 512 | 82.6% | 33.3% |
| PCA, strict split | 384 | 81.8% | 50.0% |

> **Strict split:** PCA fitting examples are held separate from classifier training examples. This is a cleaner evaluation of whether the compression basis generalizes.

### Held-Out Category Stress Test

PCA is fit without seeing the `Sci/Tech` category, then evaluated on the full AG News test set. Preliminary results suggest the learned subspace can still transfer meaningfully to the excluded category, which may indicate topic-agnostic structure in BERT's embedding space.

---

## Research Direction

Prior work has studied embedding compression and dimensionality reduction. The current direction is narrower:

> *Does a compression basis learned on distribution A remain useful for distribution B?*

The project studies transfer across:

- held-out categories, such as PCA without `Sci/Tech` evaluated on `Sci/Tech`;
- shifted datasets, such as AG News to SST-2;
- different NLP tasks;
- different embedding models;
- alternative interpretable compression methods.

After advisor feedback, the next emphasis is on interpretable PCA-related methods rather than learned black-box compression. The newest experiment compares:

- standard PCA, using covariance eigenvectors;
- sparse precision eigenbasis, using graphical Lasso to estimate a sparse inverse covariance matrix;
- random projection, as a fast non-adaptive baseline.

The sparse precision matrix can be interpreted as a graph over embedding dimensions, where nonzero entries represent conditional dependencies between variables.

---

## Repo Structure

```text
bert-embedding-compression/
├── outputs/
│   ├── bert_compression_research_after_meeting.ipynb   # Main executed experiment notebook
│   ├── bert_compression_research_graph_based.ipynb     # Next graph-based compression notebook
│   ├── bert_compression_summary.tex                    # Short LaTeX summary for advisor
│   └── graph_based_next_steps.md                       # Advisor-guided next-step plan
├── .gitignore
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/YujunGe07/bert-embedding-compression.git
cd bert-embedding-compression
pip install transformers scikit-learn datasets torch numpy pandas matplotlib
```

Open `outputs/bert_compression_research_after_meeting.ipynb` in Jupyter or Colab to reproduce the current executed experiments.

Open `outputs/bert_compression_research_graph_based.ipynb` to run the next sparse precision / graph-based compression comparison.

---

## Planned Next Steps

- [ ] Run the sparse precision basis experiment.
- [ ] Evaluate the sparse precision basis under the held-out `Sci/Tech` stress test.
- [ ] Cross-task transfer: AG News to SST-2.
- [ ] Compare transferred PCA against in-task PCA.
- [ ] Compare PCA, sparse precision eigenbasis, and random projection.
- [ ] Explore graph-learning methods from STAC-USC if the simple graphical Lasso baseline looks promising.
- [ ] Repeat across multiple random seeds.
- [ ] Measure whether PCA subspace similarity or variance explained predicts transfer accuracy.

---

## Background

BERT produces 768-dimensional `[CLS]` embeddings. PCA finds a lower-dimensional linear subspace that captures maximum variance. The core research question is whether this subspace is task-specific or captures more general structure of language, and whether this can enable efficient embedding reuse across applications.

Graph-based variants may provide more interpretable compression bases by estimating sparse dependencies between embedding dimensions.

---

## Status

Incomplete preliminary research project. Not publication-ready yet.

---

## Author

**Yujun Ge** · [GitHub](https://github.com/YujunGe07) · [Email](mailto:geyujunamy@gmail.com)
