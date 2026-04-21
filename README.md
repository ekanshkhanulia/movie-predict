# Assignment 2: SASRec (Sequential Recommendation)

## Objective

This project implements **SASRec** (Self-Attentive Sequential Recommendation) in PyTorch to predict the next movie a user will interact with based on past behavior.

The assignment goals are to:
- implement SASRec from scratch (core architecture and training logic),
- train/evaluate on **MovieLens 1M**,
- report **Recall@10**, **Recall@20**, **NDCG@10**, and **NDCG@20**,
- compare model configurations (blocks, hidden size, heads, max sequence length).

Official reference paper/code:
- [SASRec GitHub (kang205)](https://github.com/kang205/SASRec)

---

## Dataset

Use MovieLens 1M ratings with format:

`userId::movieId::rating::timestamp`

Preprocessing rules:
- treat ratings `>= 4` as positive interactions,
- discard ratings `< 4`,
- sort each user's interactions chronologically by timestamp,
- keep users with at least 5 positive interactions.

---

## Project Structure

- `data.py`: MovieLens loading, preprocessing, filtering, leave-one-out splits, dataloaders.
- `model.py`: SASRec architecture (embeddings, self-attention blocks, masking, prediction layers, loss).

---

## 1) Data Preprocessing (Rubric: 15 pts)

Implemented in `MovieLensDataset` (`data.py`):

- **Loading**:
  - reads `movies.dat` and `ratings.dat`,
  - converts `::` delimiter to a tab separator for faster parsing.
- **Implicit conversion**:
  - converts ratings to binary (`1` for `>=4`),
  - keeps only positives.
- **Chronological sequences**:
  - sorts interactions by `Timestamp`,
  - groups by user to create per-user ordered item sequences.
- **User filtering**:
  - removes users with fewer than 5 interactions.
- **Leave-one-out split**:
  - training prefix: all but last two items,
  - validation target: second-to-last item,
  - test target: last item.
- **Sequence formatting**:
  - left-padding with `0` and truncation to `maxlen`,
  - training labels are shifted next-item targets for each timestep,
  - stores user history sets for negative sampling.

---

## 2) SASRec Model (Rubric: 20 pts)

Implemented in `SASRec` (`model.py`):

- **Embedding layer**:
  - item embedding table with `padding_idx=0`,
  - learnable positional embeddings.
- **Transformer-style blocks**:
  - multi-head self-attention (`nn.MultiheadAttention`),
  - point-wise feedforward network,
  - residual connections and layer normalization (Pre-LN style),
  - dropout for regularization.
- **Causal masking**:
  - upper-triangular attention mask prevents future leakage.
- **Padding handling**:
  - key padding mask excludes padded tokens from attention.
- **Prediction heads**:
  - `predict_next(...)`: pointwise scoring for positive/negative sampled items,
  - `predict_all_items(...)`: full-item scoring for ranking at evaluation time.

---

## 3) Training and Optimization (Rubric: 20 pts)

### Training Objective

Use next-item prediction with binary classification on:
- one positive target item,
- one (or more) sampled negative items.

Current loss in `model.py`:
- binary cross-entropy with logits over positive and negative scores,
- mask out padded positions when averaging.

### Optimizer and Regularization

- optimizer: Adam (`torch.optim.Adam`)
- regularization:
  - dropout in embeddings/attention/FFN,
  - layer normalization in every block,
  - weight decay (already set in optimizer).

### Early Stopping

Implement early stopping using validation **NDCG@10**:
- evaluate after each epoch,
- if validation NDCG@10 does not improve for `patience` epochs, stop,
- keep best model checkpoint.

---

## 4) Evaluation (Rubric: 15 pts)

Evaluate on validation and test using sequence prefix and held-out next item.

Report:
- Recall@10
- Recall@20
- NDCG@10
- NDCG@20

### Metric Definitions

For each user, rank candidate items by predicted score:

- `Recall@K = 1` if true next item appears in top-K, else `0` (then averaged).
- `NDCG@K = 1 / log2(rank+1)` if true item rank <= K, else `0` (then averaged).

### Candidate Set

Use one of the following evaluation protocols consistently:
- **Sampled ranking**: true item + N random negatives not in user history.
- **Full ranking**: rank among all items not seen in training history.

State clearly which protocol you use in your report.

---

## Suggested Experiment Configurations

Compare at least these settings:

- number of blocks: `[1, 2, 3]`
- hidden size: `[64, 128, 256]`
- attention heads: `[1, 2, 4]`
- max sequence length: `[50, 100, 200]`

Keep other hyperparameters fixed while changing one factor at a time for fair comparison.

---

## Example Training Setup

Recommended starting hyperparameters:
- learning rate: `1e-3` (or `5e-4`)
- batch size: `128`
- dropout: `0.2`
- epochs: `50`
- patience (early stop): `5`
- negatives per positive: `1` to `5`

---

## Reproducibility Checklist

- set random seeds (`random`, `numpy`, `torch`),
- log all hyperparameters,
- save best checkpoint by validation NDCG@10,
- report both validation and test metrics,
- include hardware/environment details.

---

## What to Submit

1. Source code (`data.py`, `model.py`, training/eval script).
2. This README.
3. A short report including:
   - preprocessing pipeline,
   - model design summary,
   - training details,
   - metric results table (Recall/NDCG at 10/20),
   - hyperparameter comparison table/plot,
   - brief discussion of findings.

---

## Notes

- This repository currently contains data and model components.
- Add a `train.py` (or notebook) that wires dataset, model, negative sampling, training loop, evaluation, and early stopping for full end-to-end execution.
