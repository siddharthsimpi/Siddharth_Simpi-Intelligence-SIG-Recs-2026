# Paper notes: DLRM

*Naumov et al., "Deep Learning Recommendation Model for Personalization and Recommendation Systems", arXiv:1906.00091 (Facebook/Meta AI, 2019).
These are my own summary notes; check numbers and claims against the paper.*

## Problem and idea
Industrial recommendation / click-prediction models consume two very different kinds of input: **dense (continuous) features** and
**sparse categorical features** (user id, ad id, page, ...). DLRM is a deliberately simple reference architecture that treats them in a
principled way and is built to be benchmarked on real hardware.

## Architecture (what `ctrlib/models.py::DLRM` implements)
1. **Bottom MLP** maps the dense features to a vector of the same dimension d as the embeddings.
2. Every categorical feature has an **embedding table** (rows = categories, d columns); a lookup gives one d-dim vector per feature.
3. **Interaction layer**: take all vectors {bottom-MLP output, e_1 ... e_F} and compute the **pairwise dot products** between them
   (inspired by factorization machines: second-order interactions only). With F = 26 categorical features this is 27*26/2 = 351 products
   (self-products excluded).
4. The dense vector and the dot products are concatenated and passed through the **top MLP**; a sigmoid gives the click probability.
   Training minimises binary cross-entropy.

Why dot products: the dot product of two embeddings is a learned similarity, the same inductive bias as matrix factorization
(Task 1), but here applied between *all* pairs of fields, followed by a non-linear network that decides how to use them.

## Systems contribution (not reproduced here)
- Embedding tables are huge (up to billions of rows) and dominate memory, while the MLPs are small but compute-heavy.
- The paper therefore proposes **hybrid parallelism**: embedding tables are **model-parallel** (sharded across devices), MLPs are
  **data-parallel**, with an all-to-all communication step between the embedding lookups and the interaction layer.
- Our NumPy single-process version keeps the *model* faithful but cannot reproduce the parallel training system.

## Training / evaluation details to remember
- Original reference experiments: Criteo Ad Kaggle data and synthetic benchmarks, accuracy compared against Deep & Cross Network.
  (See the paper for the exact numbers; this repo uses a 40k-row sample, so absolute metrics are not comparable.)
- Reference implementation initialises embeddings U(-sqrt(1/n), sqrt(1/n)) (n = rows of the table); we use the same init
  (and test a small normal init as an option).

## Relation to the other models in this repo
| model | how it handles interactions |
|---|---|
| Logistic regression | none (linear in one-hot features) |
| FM (Task 1 lineage) | all pairwise dot products, but **summed** into one scalar (no non-linearity on top) |
| Vanilla MLP (Task 2) | implicit: the network must discover products from concatenated embeddings |
| DCN (Task 2 bonus) | explicit polynomial crosses of the concatenated input, bounded degree |
| DLRM (Task 3) | explicit pairwise dot products kept as **separate features** for a top MLP |

## Expected strengths and limitations (my reading)
- Strengths: clean separation of dense/sparse paths, explicit second-order structure, scales with sharded embeddings.
- Limitations: only second-order interactions, the interaction layer grows quadratically with the number of fields, all
  embeddings share one dimension d, and with little data the huge embedding tables over-fit easily.
