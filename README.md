# BERT Embedding Compression

<div align="left">

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21F?style=flat&logo=huggingface&logoColor=black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active%20Research-blue?style=flat)

</div>

> **Can a compression basis learned on one data distribution transfer to another?**
> This project investigates PCA-based dimensionality reduction of BERT embeddings, with a focus on *cross-distribution transfer* — whether subspaces learned on held-out categories, shifted datasets, and different NLP tasks remain meaningful.

---

## Results

### AG News Classification (BERT `[CLS]` embeddings → Logistic Regression)

| Setting | Dimensions | Accuracy | Reduction |
|---------|-----------|----------|-----------|
| Baseline | 768 | 83.0% | — |
| PCA (joint split) | 128 | **83.4%** | **83.3%** |
| PCA (strict split) | 512 | 82.6% | 33.3% |
| PCA (strict split) | 384 | 81.8% | 50.0% |

> **Strict split**: PCA fit examples are held entirely separate from classifier training examples — a cleaner evaluation of whether the compression basis generalizes.

### Held-Out Category Stress Test
PCA basis fit *without* seeing the `Sci/Tech` category, then evaluated on the full AG News test set. Results indicate the subspace learned from 3 categories transfers meaningfully to the excluded one — suggesting topic-agnostic structure in BERT's embedding space.

---

## Research Direction

Prior work has studied BERT embedding compression, but **cross-distribution transfer** of the compression basis remains underexplored. The key question:

> *Does a PCA subspace learned on distribution A remain a good basis for data from distribution B?*

We study transfer across:
- **Held-out categories** (e.g., PCA without Sci/Tech → evaluated on Sci/Tech)
- **Shifted datasets** (e.g., AG News → SST-2)
- **Different NLP tasks** (classification → entailment / QA)
- **Different embedding models** (BERT-base → RoBERTa, DistilBERT)
- **Alternative compression methods** (random projection, autoencoders, quantization)

---

## Repo Structure

```
bert-embedding-compression/
├── outputs/
│   ├── bert_compression_research_after_meeting.ipynb   # Main experiment notebook
│   └── bert_compression_summary.tex                    # LaTeX summary for advisor
├── .gitignore
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/YujunGe07/bert-embedding-compression.git
cd bert-embedding-compression
pip install transformers scikit-learn datasets torch numpy
```

Open `outputs/bert_compression_research_after_meeting.ipynb` in Jupyter to reproduce experiments.

**Dependencies:** `transformers`, `scikit-learn`, `datasets` (HuggingFace), `torch`, `numpy`

---

## Planned Next Steps

- [ ] Cross-task transfer: AG News → SST-2
- [ ] Compare transferred PCA vs. in-task PCA baselines
- [ ] Add random projection, autoencoder, and quantization baselines
- [ ] Multi-seed variance analysis
- [ ] Measure whether PCA subspace similarity (e.g., subspace angle) predicts transfer accuracy

---

## Background

BERT produces 768-dimensional `[CLS]` embeddings. PCA finds a lower-dimensional linear subspace that captures maximum variance. The question is whether this subspace is *task-specific* or encodes more *general* structure of language — and whether knowing this can enable efficient embedding reuse across applications.

---

## Author

**Yujun Ge** · [GitHub](https://github.com/YujunGe07) · [Email](mailto:geyujunamy@gmail.com)
