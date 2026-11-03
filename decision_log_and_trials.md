# Chest-CT-Scan-Classifier: Trials and Decision Log

Last updated: 2026-11-03. Test set: 269 images (120 adenocarcinoma, 51 large_cell, 8 normal, 90 squamous_cell). Random guess loss is about 1.39. Always-guess-adenocarcinoma accuracy is 44.6%.

## 1. Trials so far

| # | What we changed | Train / Val acc | Test acc | Test loss | Macro F1 |
|---|---|---|---|---|---|
| R0 | Flatten, LR 1e-3 (not logged) | ~80% / ~80%, jagged | not measured | n/a | n/a |
| R1 | Flatten, LR 1e-4 (baseline) | ~77% / ~73% | 69.9% | 2.04 | ~0.69 |
| R2 | GAP head, LR 1e-4 | 57% / 49% | 37.2% | 1.79 | n/a |
| R3 | GAP head, LR 1e-3 | ~75% / ~74% | 50.9% | 1.19 | n/a |
| R4 | Flatten + dropout 0.5 (assumed), LR 1e-4 | ~62% / ~61% | 63.2% | 2.98 | 0.666 |
| R5 | GAP + dropout 0.3, LR 1e-3 | ~71% / ~74% | 59.1% | 0.97 | 0.653 |
| R6 | R5 + unfreeze `block5_conv1` onwards, LR 1e-5 | ~93% / ~87% | 69.1% | 0.68 | 0.651 |

Note: R6 accuracy (69.1%) is about the same as R1 (69.9%), a 2-image difference. The real gain is test loss (2.04 to 0.68) and adenocarcinoma recall.

## 2. Diagnosis and lesson per trial

| # | Diagnosis | Lesson |
|---|---|---|
| R0 | Unstable training | LR 1e-3 is too big for the Flatten head (loss starts around 16) |
| R1 | Mild underfitting, overconfident | Reliable baseline; huge head makes the loss start high |
| R2 | Severe underfitting | Small head + small LR learns too slowly |
| R3 | Healthy training, capped | After changing the head, the best LR changes too |
| R4 | Over-regularized | Dropout did not help; the problem was not memorizing |
| R5 | Underfitting (train ~ val) | Smooth and calibrated, but frozen features are the limit |
| R6 | Starting to overfit (93% vs 87%) | Unfreezing broke the cap; variance is now the issue |

## 3. Fitting cheat sheet

| Case | Train | Val | Gap | Fix direction |
|---|---|---|---|---|
| Underfitting | low | low | small | More capacity, higher LR, train longer, unfreeze |
| Healthy | high | close behind | small | Keep going |
| Overfitting | very high | lower, flattening | large | Augmentation, dropout, weight decay, more data |
| Unstable | jagged, loss rising | jagged | any | Lower LR |
| Split mismatch | n/a | high | val much greater than test | Fix splits or evaluation, not the model |

Validation (30% large_cell, 16% normal) does not match test (19% large_cell, 3% normal), so val accuracy flatters a model that over-predicts large_cell.

## 4. Where capacity is added matters

| What you add | What it changes | Effect |
|---|---|---|
| Bigger head (Flatten, extra Dense) | Re-weights existing features only | Train acc up, overfits easily, rarely helps test |
| Unfreezing backbone layers | Changes the features themselves | Helps when features are the bottleneck |
| GAP instead of Flatten | Removes head capacity | More stable, less overfitting, slower learning |

## 5. Knobs and their effects

| Knob | Train acc | Val / test acc | Train-val gap | Use when |
|---|---|---|---|---|
| Learning rate down | slower, steadier | steadier | no change | Loss jagged or rising |
| Learning rate up | faster | may jump | may grow | Learning too slowly |
| Epochs / patience up | up | up then down | grows | Curves still falling |
| Unfreeze more layers | much up | up if features limit | up | Train ~ val, both low |
| Freeze more layers | down | steadier | down | Overfitting on tiny data |
| Bigger head / Dense | up | rarely helps | up | Only if head is the limit |
| GAP instead of Flatten | down | steadier | down | Head overfits or loss starts huge |
| Dropout up | slightly down | up | down | Gap is large |
| Weight decay (L2) | slightly down | up | down | Gap is large |
| Augmentation up | slightly down | up | down | Gap is large |
| More / better data | about same | much up | much down | Almost always the best fix |
| Batch size down | about same | slight regularizer | down | Overfitting; VRAM limits |
| LR schedule | up at the end | steadier | down | Curves bounce late |
| Class weights | may go down | per-class recall shifts | about same | A class is rare and ignored |
| Better backbone | up | up | varies | Everything else tried |
| Label smoothing | about same | about same | about same | Overconfident predictions |

## 6. Planned trials (one change at a time, fixed seed)

| # | Change | Why | Expect (train / val / test / loss) | Keep if |
|---|---|---|---|---|
| T0 | Repeat R6 with 2-3 seeds | R1 vs R6 differ by 2 images | n/a | You know the noise level |
| T1 | Stronger augmentation | R6 gap 93% vs 87% | train down a bit, val same or up, test up?, loss down | Test macro F1 up and gap down |
| T2 | Dropout 0.3 to 0.5 and/or L2 on head | Same gap | train down slightly, test about same | Gap down, test not down |
| T3 | Fine-tune LR 1e-5 to 3e-5, or ReduceLROnPlateau | Late curves bounce | loss down slightly | Val curve smoother |
| T4 | Class weights, or stratified val re-split | large_cell over-predicted; val differs from test | large_cell precision up, recall down | Macro F1 up |
| T5 | Unfreeze `block4_conv1` | Only if T1-T2 leave train too low | train much up, gap up | Test up, else revert |
| T6 | ResNet50 or EfficientNetB0, same recipe | Backbone comparison | varies | Macro F1 up |

Why not unfreeze more yet: the bottleneck moved from underfitting to generalization, so regularize first.

How to read a result:

| Train | Val | Test | Meaning | Action |
|---|---|---|---|---|
| down small | same or up | up | Good trade | Keep |
| up | down | down | More overfitting | Revert |
| down big | down | down | Over-regularized | Back off |
| same | same | same | No effect | Revert |

## 7. Decision log

| # | Decision | Why | Outcome |
|---|---|---|---|
| 1 | Use VGG16 | Tutorial baseline; simple to freeze and inspect | Worked; compare other backbones later (T6) |
| 2 | `preprocess_input`, no 1/255 rescale | VGG16 expects its own scaling | Correct |
| 3 | Monitor `val_loss`, patience 5, restore best weights | Small data; keep the best epoch | Stops sensibly |
| 4 | Drop `steps_per_epoch` / `validation_steps` | Floor division skipped images | All images used |
| 5 | Evaluate on test with `shuffle=False` | Predictions must match labels | Honest score |
| 6 | LR 1e-3 to 1e-4 (Flatten head) | R0 was unstable | R1 baseline |
| 7 | Try GAP | Loss started at ~8; ~100k head weights | Calmer, but slow at 1e-4 (R2) |
| 8 | GAP with LR 1e-3 | R2 was underfitting | Smooth (R3) |
| 9 | Put `HEAD_TYPE`, `DROPOUT_RATE` in params.yaml | MLflow could not see architecture changes | Runs are labeled |
| 10 | Add classification report and confusion matrix | Accuracy hid class errors | Found the large_cell bias |
| 11 | Flatten + dropout 0.5 | Test whether the head overfits | No help (R4) |
| 12 | Check class counts per split | Explain the bias | Not caused by train mix; val differs from test |
| 13 | Skip the validation re-split | Data already arrives pre-split | Validation stays unreliable (known) |
| 14 | Unfreeze `block5`, LR 1e-5 | Three heads capped at 60-70% with no gap | Loss 0.97 to 0.68, train 71% to 93% |
| 15 | Add DVC pipeline | Reproducibility; avoid stale params | Done |

## 8. Working rules

- Change one thing per run; log it in MLflow via params.yaml.
- Run stages 03, 04, 05 together after an architecture change (or use `dvc repro`).
- Pick models by validation and macro F1; use test only to report.
- Compare accuracy against the always-guess-adenocarcinoma baseline (44.6%).
- Small differences (about 3% or less) are noise on a 269-image test set and a 70-image val set.
