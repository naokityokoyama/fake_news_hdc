# Hyperdimensional Computing for Fake News Detection

**A High-Efficiency Approach for Large-Scale Data**

Naoki Yokoyama · Leandro Santiago de Araújo
Instituto de Computação — Universidade Federal Fluminense (UFF), Niterói, RJ, Brazil

This repository contains the code for the paper *Hyperdimensional Computing for Fake News Detection: A High-Efficiency Approach for Large-Scale Data*, part of the author's master's research at PPGC/IC-UFF.

---

## Overview

State-of-the-art fake news detection is dominated by Transformers and Large Language Models, which deliver high accuracy at a high computational and energy cost. This work investigates **Hyperdimensional Computing (HDC)** as an alternative paradigm and frames the problem as a **trade-off between accuracy and energy efficiency**.

We evaluate five HDC variants against classical Machine Learning baselines and Transformers (BERT, RoBERTa) on three benchmarks — ISOT, COVID-19 and a binarized FEVER — totaling **160,051 instances**.

**Key findings**

- Transformers achieve the highest F1-score on all three datasets (99.80% on ISOT with BERT).
- The best HDC configuration reaches **95.77% F1 on ISOT**, about 4 points below BERT.
- HDC reduces training time — and the corresponding TDP-based energy estimate — by **≈5.5× to ≈13×** relative to BERT.
- HDC is not a drop-in replacement when maximum predictive performance is the goal, but it is a promising **efficiency-oriented alternative** for large-scale and resource-constrained misinformation detection, e.g. as a low-cost first-line filter whose positive flags are confirmed by a Transformer.

---

## Repository structure

| File | Description |
| --- | --- |
| [`torchhd.ipynb`](torchhd.ipynb) | HDC classifiers (Vanilla HDC, AdaptHD, OnlineHD, NeuralHD, DistHD) built with the [torchhd](https://github.com/hyperdimensional-computing/torchhd) library |
| [`ML_TFIDF_W2V_BOW.ipynb`](ML_TFIDF_W2V_BOW.ipynb) | Text representations (Bag-of-Words, TF-IDF, Word2Vec) and the classical ML baselines |
| [`FAKENEWS_HDC_ML_ngram.ipynb`](FAKENEWS_HDC_ML_ngram.ipynb) | Classical ML models using the N-Gram encoder |
| [`Fake_news_Bert_Roberta.ipynb`](Fake_news_Bert_Roberta.ipynb) | Fine-tuning of the Transformer baselines (BERT and RoBERTa) |

All notebooks were developed and run on Google Colab.

---

## Datasets

The three datasets were chosen to isolate distinct stress variables: content learning under balanced conditions (ISOT), minority-class detection under class imbalance (COVID-19), and scalability in large-scale, context-dependent scenarios (FEVER).

| Dataset | Total | Real | Fake | Test (30%) | Characteristics |
| --- | ---: | ---: | ---: | ---: | --- |
| **ISOT** | 44,267 | 21,416 | 22,851 | 13,281 | Nearly balanced, long full-length articles |
| **COVID-19** | 5,975 | 4,532 | 1,443 | 1,793 | Imbalanced (≈3:1), short posts and snippets |
| **FEVER** (binarized) | 109,809 | 80,034 | 29,775 | 32,943 | Large-scale, short Wikipedia-derived claims (≈2.7:1) |
| **Total** | **160,051** | 105,982 | 54,069 | | |

- **ISOT** — [Ahmed et al., 2017]. Articles from verified agencies (e.g., Reuters) and from sources flagged as unreliable by fact-checkers.
- **COVID-19** — A consolidated collection in the lineage of [Patwa et al., 2021]. Unlike the original, roughly balanced 10,700-post release, this version is markedly imbalanced, which is exploited here as a minority-class stress test.
- **FEVER (binarized)** — Built from the original training partition of [Thorne et al., 2018] (145,449 claims):
  1. `NOT ENOUGH INFO` is discarded (no equivalent in the real/fake dichotomy);
  2. `SUPPORTS → real`, `REFUTES → fake`;
  3. exact duplicates are removed via SHA-256 hashing of the normalized text;
  4. the stratified 70/30 split is performed **after** deduplication, guaranteeing zero overlap between partitions.

  Models see the **claim text only**, without the evidence-retrieval stage, so results are **not comparable** to full FEVER-Score systems. This setting should be read as a deliberately harder, text-only scalability stress test.

The raw datasets are not redistributed here; please obtain them from the original sources.

---

## Methodology

### Preprocessing

A uniform pipeline is applied to all datasets and to all non-Transformer models:

1. Removal of digital artifacts (URLs, HTML tags, mentions, hashtags)
2. Lowercasing
3. Stopword filtering
4. Porter stemming

Stemming benefits HDC twice: it increases semantic density and shrinks the Item Memory.

### Text representations

| Representation | Configuration |
| --- | --- |
| Bag-of-Words (BoW) | Vocabulary limited to 10,000 terms |
| TF-IDF | Vocabulary limited to 10,000 terms |
| Word2Vec | 300-dimensional embeddings |
| N-Gram | Native trigram-permutation encoding over the token stream |

### Hyperdimensional encoding

An **Item Memory** maps each unique token to a static atomic hypervector $\vec{H} \in \{0,1\}^D$. HDC relies on three operations: **binding** ($\otimes$, element-wise XOR), **bundling** ($\oplus$, addition followed by majority thresholding) and **permutation** ($\Pi$, cyclic rotation).

**Token-based representations (BoW, TF-IDF)** capture local syntax through trigram permutation encoding:

$$
\vec{H}_{tri} = \Pi^2(\vec{H}_{w_1}) \otimes \Pi(\vec{H}_{w_2}) \otimes \vec{H}_{w_3}
$$

and the document hypervector bundles all trigrams, with TF-IDF weights (when present) scaling each trigram's contribution:

$$
\vec{H}_{doc} = \mathrm{threshold}\Big(\sum_i \vec{H}_{tri,i}\Big)
$$

**Word2Vec** has no token-level correspondence, so each component $j$ of the document vector $\vec{v} \in \mathbb{R}^{300}$ is bound to a random position hypervector $\vec{P}_j$ and a quantized level hypervector $\vec{L}(v_j)$:

$$
\vec{H}_{doc} = \mathrm{threshold}\Big(\sum_{j=1}^{300} \vec{L}(v_j) \otimes \vec{P}_j\Big)
$$

as implemented in torchhd [Heddes et al., 2023].

**Dimensionality.** Two values were evaluated, $D \in \{1{,}000;\ 3{,}000\}$. The usual literature reference $D = 10{,}000$ was not feasible for the full experimental grid on the available hardware, since memory and per-epoch cost grow linearly with $D$. The main experiments use $D = 3{,}000$.

### Models

| Family | Models |
| --- | --- |
| **HDC** | Vanilla HDC, AdaptHD, OnlineHD, NeuralHD, DistHD |
| **Classical ML** | Logistic Regression, SVM, Random Forest |
| **Transformers** | BERT, RoBERTa |

- **AdaptHD** [Imani et al., 2019] — perceptron-style iterative error correction of class prototypes.
- **OnlineHD** [Hernández-Cano et al., 2021] — online learning with novelty-dependent update magnitudes.
- **NeuralHD** [Zou et al., 2021] — regenerates poorly informative dimensions during training; non-linear sinusoidal encoding.
- **DistHD** [Wang et al., 2023] — learner-aware dynamic encoding with a top-2 decision rule.

### Experimental protocol

| Parameter | Value |
| --- | --- |
| Train/test split | 70/30, stratified, `random_state = 42` |
| HDC dimensionality | 1,000 and 3,000 |
| Adaptive HDC retraining epochs | 2 (other hyperparameters at reference-implementation defaults) |
| Classical ML | scikit-learn defaults |
| BERT / RoBERTa | 2 epochs, batch size 16, learning rate 2e-5, `max_length` 512 |
| Primary metric | F1-score of the positive class (**fake**) |

On COVID-19, F1 is the primary metric: a trivial majority-class classifier already reaches ≈75.85% accuracy with a null fake-class F1.

### Energy estimation

Energy is estimated from the GPU's nominal Thermal Design Power:

$$
E\,[\mathrm{Wh}] = \frac{\mathrm{TDP} \times t}{3600}
$$

where $t$ is the training time in seconds and TDP = 70 W (NVIDIA Tesla T4). These are **theoretical upper-bound estimates**, since they assume continuous operation at maximum power. Because TDP is constant, energy ratios between models are, by construction, identical to their training-time ratios.

### Environment

- Google Colab Pro — NVIDIA Tesla T4 (16 GB VRAM), 12 GB RAM, Intel Xeon, 2 vCPUs
- Main libraries: `torchhd`, `scikit-learn`, Hugging Face `transformers`

---

## Results

### Best HDC configuration per dataset (D = 3,000)

| Dataset | Best HDC | Representation | Accuracy (%) |
| --- | --- | --- | ---: |
| ISOT | AdaptHD | Bag-of-Words | 95.6 |
| COVID-19 | NeuralHD | TF-IDF | 86.5 |
| FEVER | OnlineHD | Bag-of-Words | 78.0 |

Classical baselines remain strong (e.g., Random Forest reaches 96.3% accuracy on ISOT). No single text representation is optimal across all datasets.

### Effect of hypervector dimensionality

| Dataset | Model | Representation | D | Acc. (%) | F1 (%) |
| --- | --- | --- | ---: | ---: | ---: |
| ISOT | AdaptHD | BoW | 1,000 | 87.68 | 88.11 |
| | AdaptHD | BoW | 3,000 | 95.56 | **95.77** |
| COVID-19 | NeuralHD | TF-IDF | 1,000 | 85.95 | 72.43 |
| | NeuralHD | TF-IDF | 3,000 | 86.50 | **73.16** |
| FEVER | OnlineHD | BoW | 1,000 | 77.35 | **50.52** |
| | OnlineHD | BoW | 3,000 | 78.00 | 50.28 |

Higher dimensionality helps substantially on ISOT, modestly on COVID-19, and not at all on FEVER: its impact is dataset-dependent.

### Transformers vs. HDC

| Dataset | Model | Acc. (%) | F1 (%) | Time (s) | Energy (Wh) | Reduction vs. BERT |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **ISOT** | BERT | 99.79 | **99.80** | 4,357.80 | 84.73 | — |
| | RoBERTa | 99.73 | 99.74 | 4,450.00 | 86.53 | — |
| | AdaptHD (BoW) | 95.56 | 95.77 | 796.35 | 15.48 | **≈5.5×** |
| **COVID-19** | BERT | 91.47 | **83.24** | 123.60 | 2.40 | — |
| | RoBERTa | 90.96 | 81.55 | 126.50 | 2.46 | — |
| | NeuralHD (TF-IDF) | 86.50 | 73.16 | 9.37 | 0.18 | **≈13×** |
| **FEVER** | BERT | 85.99 | **70.09** | 2,532.40 | 49.24 | — |
| | RoBERTa | 85.41 | 69.35 | 2,585.00 | 50.26 | — |
| | OnlineHD (BoW) | 78.00 | 50.28 | 447.04 | 8.69 | **≈5.7×** |

- Transformers retain the best F1 everywhere, with RoBERTa consistently within one point of BERT.
- On balanced ISOT, HDC stays about 4 F1 points behind BERT at ≈5.5× lower cost. The gap grows to ≈10 points on COVID-19 and ≈29 points on FEVER.
- The low absolute F1 of all models on FEVER, BERT included, reflects the difficulty of claim-only verification.
- On COVID-19, NeuralHD's fake-class prototype (built from only 1,443 training vectors) reaches 77.46% recall at 69.33% precision (146 FP vs. 96 FN): it catches most fake posts but over-flags legitimate ones. This profile supports using HDC as a **first-line filter** with Transformer confirmation on ambiguous cases.

### Note on the N-Gram results

On ISOT, every model with the N-Gram representation — classical baselines included — collapses to within one point of the majority-class baseline (51.62%), whereas the literature reports ≈92% with n-gram features. These results are likely affected by an implementation issue and are reported **for transparency only**. On the deduplicated FEVER split, the best N-Gram result drops to 65.40% (DistHD), suggesting that high scores previously reported on non-deduplicated corpora may reflect train/test overlap rather than the effectiveness of the encoding.

---

## Limitations

- The N-Gram encoding pipeline is faulty on ISOT (see above).
- HDC models use textual representations only, with no external evidence, which limits reasoning-heavy fact-checking.
- Energy is estimated from TDP rather than measured directly, and only **training** cost was quantified; inference latency and energy remain to be characterized.
- All experiments ran on a single platform (Tesla T4); edge, FPGA and neuromorphic benefits were not validated.
- Dimensionality was restricted to $D \le 3{,}000$ by memory constraints.
- A single random seed precludes significance testing.
- ISOT real-class articles come predominantly from Reuters, so classifiers may capture source-style markers rather than veracity.

## Future work

Hybrid HDC–Transformer frameworks; Item Memory initialization from pretrained embeddings; multilingual encoders; comparison with compressed Transformers (DistilBERT, TinyBERT); multi-seed stability analysis with confidence intervals; $D = 10{,}000$; inference latency and energy measurement; FPGA and neuromorphic deployment; direct energy measurement via NVML or CodeCarbon.

---

## Reproducing the experiments

1. Open the notebooks in Google Colab (a GPU runtime such as the Tesla T4 is recommended).
2. Install the dependencies:

   ```bash
   pip install torch torch-hd scikit-learn transformers gensim nltk pandas numpy matplotlib
   ```

   > The PyPI package is named `torch-hd`; it is imported as `import torchhd`.

3. Download the ISOT, COVID-19 and FEVER datasets from their original sources and update the data paths in each notebook.
4. Run the notebooks:
   - `ML_TFIDF_W2V_BOW.ipynb` — text representations and classical ML baselines
   - `FAKENEWS_HDC_ML_ngram.ipynb` — N-Gram experiments
   - `torchhd.ipynb` — HDC variants at $D \in \{1{,}000;\ 3{,}000\}$
   - `Fake_news_Bert_Roberta.ipynb` — BERT and RoBERTa fine-tuning

All experiments use `random_state = 42`.

---

## Citation

If you use this code, please cite:

```bibtex
@misc{yokoyama2026hdcfakenews,
  author = {Yokoyama, Naoki and Ara{\'u}jo, Leandro Santiago de},
  title  = {Hyperdimensional Computing for Fake News Detection:
            A High-Efficiency Approach for Large-Scale Data},
  year   = {2026},
  note   = {Instituto de Computa{\c{c}}{\~a}o, Universidade Federal Fluminense},
  howpublished = {\url{https://github.com/naokityokoyama/fake_news_hdc}}
}
```

## References

- Ahmed, H., Traore, I., Saad, S. (2017). Detection of online fake news using n-gram analysis and machine learning techniques. *ISDDC*, Springer.
- Devlin, J. et al. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL-HLT*.
- Heddes, M. et al. (2023). Torchhd: An open source Python library to support research on hyperdimensional computing and vector symbolic architectures. *JMLR*, 24(255).
- Hernández-Cano, A. et al. (2021). OnlineHD: Robust, efficient, and single-pass online learning using hyperdimensional system. *DATE*.
- Imani, M. et al. (2019). AdaptHD: Adaptive efficient training for brain-inspired hyperdimensional computing. *BioCAS*.
- Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. *Cognitive Computation*, 1.
- Liu, Y. et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv:1907.11692*.
- Patwa, P. et al. (2021). Fighting an infodemic: COVID-19 fake news dataset. *CONSTRAINT*, Springer.
- Thorne, J. et al. (2018). FEVER: A large-scale dataset for fact extraction and VERification. *NAACL-HLT*.
- Wang, J., Huang, S., Imani, M. (2023). DistHD: A learner-aware dynamic encoding method for hyperdimensional classification. *DAC*.
- Zou, Z. et al. (2021). Scalable edge-based hyperdimensional learning system with brain-like neural adaptation. *SC21*.

## License

To be defined.

## Acknowledgments

To Prof. Dr. Leandro Santiago de Araújo for the supervision, and to the Instituto de Computação at UFF.
