# ML Strategy — Study Notes

---

## 1. Ideas for Improving a Model

When a model is underperforming, here are the main levers to pull:

| # | Strategy | Targets |
|---|----------|---------|
| 1 | Collect more data | Variance |
| 2 | Collect more diverse training data | Variance / Distribution |
| 3 | Train longer with gradient descent | Bias |
| 4 | Try Adam instead of vanilla GD | Bias / Optimisation |
| 5 | Try a bigger network | Bias |
| 6 | Try a smaller network | Variance |
| 7 | Add Dropout | Variance |
| 8 | Add L2 regularisation | Variance |
| 9 | Tune architecture (activations, hidden units) | Bias / Variance |

---

## 2. Orthogonalisation

> A design principle ensuring that tuning one hyperparameter or component affects **only one specific aspect** of performance, without unintended side effects on others.

### Chain of Assumptions in ML

Each stage must be solved before the next matters:

```
1. Fit training set well on cost function         →  fix: bigger model, better optimiser
2. Fit dev set well on cost function              →  fix: regularisation, more data
3. Fit test set well on cost function             →  fix: bigger dev set
4. Performs well in the real world                →  fix: change dev/test set or metric
```

---

## 3. Evaluation Metrics

### Classification Metrics

| Metric | Formula | Focus |
|--------|---------|-------|
| **Precision** | TP / (TP + FP) | Minimise false positives |
| **Recall** | TP / (TP + FN) | Minimise false negatives |
| **F1 Score** | 2 · (P · R) / (P + R) | Balance of both |

> F1 is especially useful for **imbalanced datasets** — accuracy alone can be misleading.

---

## 4. Satisficing vs Optimising Metrics

When choosing between models, separate your metrics into two roles:

| Role | Meaning | Example |
|------|---------|---------|
| **Optimising** | Maximise / minimise as much as possible | Accuracy |
| **Satisficing** | Just needs to meet a threshold | Running time < 100 ms |

### Example

| Model | Accuracy | Running Time |
|-------|----------|-------------|
| A | 90% | 80 ms |
| B | 92% | 95 ms ✅ |
| C | 95% | 1500 ms ❌ |

Model C is eliminated because it fails the satisficing threshold (>100 ms). Among the remaining, Model B wins on the optimising metric (accuracy).

---

## 5. Train / Dev / Test Distributions

- Dev and test sets must come from the **same distribution** — they represent the data you expect in the real world.
- Choose dev/test sets to reflect the problem you actually care about.

### Recommended Splits

| Dataset Size | Split |
|-------------|-------|
| Small (thousands) | 70 / 30 or 60 / 20 / 20 |
| Large (≥ 1 million) | 98 / 1 / 1 (1% = ~10,000 examples, sufficient for validation) |

> Sometimes having no test set is acceptable — a train/dev split alone may be enough for some projects.

---

## 6. When to Change Metrics or Dev/Test Sets

If the current metric or dataset no longer reflects the real goal, change them.

**Example:**

| Model | Error | Policy Compliance |
|-------|-------|-------------------|
| A | 3% | ❌ Violates privacy policy |
| B | 5% | ✅ Follows policy |

Model B should be preferred despite higher error. The metric needs to capture this.

### Weighted Error

Add a weight term to the cost function to penalise undesired behaviour:

```
Error = Σ w(i) · L(ŷ, y) / m

where:
  w = 1   if example follows correct policy
  w = 10  if example violates policy
```

> A high weight dramatically increases the error for policy-violating examples, steering the optimiser away from them.

---

## 7. Bayes Optimal Error & Human-Level Performance

- **Bayes Optimal Error** — The theoretical best possible error any model can achieve (irreducible error due to noise in data).
- Human-level performance is used as a **practical proxy** for Bayes error.
- Over time, as models grow and train on more data, performance approaches but **never surpasses** Bayes error.

---

## 8. Avoidable Bias

The gap between **training error** and **Bayes / human-level error**.

```
Human error       : 8.0%
Training error    : 7.5%
─────────────────────────
Avoidable bias    : 0.5%   ← very little room to improve on bias
```

If training error is already close to human-level, focus shifts to **reducing variance** (closing the train → dev gap), not bias.

### Bias–Variance Diagnostic

```
Human-Level Error
      ↕  Avoidable Bias         → fix with: bigger model, better optimiser, architecture search
Training Error
      ↕  Variance               → fix with: more data, regularisation, dropout
Dev Error
```

---

## 9. Correcting Incorrect Dev/Test Labels

- Apply the **same correction process** to both dev and test sets to keep their distributions aligned.
- Review examples the algorithm **got right** as well as ones it got wrong — mislabelled correct predictions can mask real errors.
- Be aware that after correction, train and dev/test data may come from slightly different distributions.

---

## 10. Addressing Data Mismatch

When training and dev/test sets come from different distributions:

1. **Manual error analysis** — Identify specific ways the training set differs from the dev/test set.
2. **Make training data more similar** — Use data augmentation or synthesis (e.g., add background noise to audio).
3. **Collect more data** that resembles the dev/test distribution.

---

## 11. Transfer Learning

Re-use a model trained on a large dataset for a related task with less data.

```
Pre-trained model (e.g., ImageNet)
        ↓
Remove output layer
        ↓
Add new output layer for your task
        ↓
Fine-tune
```

| Your data size | Recommended approach |
|---------------|----------------------|
| **Small** | Freeze all layers; retrain only the last output layer |
| **Large** | Retrain all weights (fine-tuning) |

- **Pre-training** — Initial training on the large source dataset to initialise weights.
- **Fine-tuning** — Continued training on the smaller target dataset, updating all weights.

> Transfer learning makes sense when the low-level features of the source task (e.g., edges, textures) are useful for the target task (e.g., radiology images).

---

## 12. Multi-Task Learning

Train a single model to solve multiple tasks simultaneously, sharing lower-level representations.

**When it makes sense:**

- Tasks share useful low-level features (e.g., object detection for pedestrians, cars, and signs simultaneously).
- Each task has a roughly similar amount of data.
- You can afford a network large enough to do well on **all** tasks at once.

> Unlike transfer learning (sequential), multi-task learning trains all tasks **in parallel** within one network.

---



