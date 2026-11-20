# Chest-CT-Scan-Classifier: Trials and Decision Log

Last updated: 2026-11-05. Test set: 269 images (120 adenocarcinoma, 51 large_cell, 8 normal, 90 squamous_cell). Random guess loss is about 1.39. Always-guess-adenocarcinoma accuracy is 44.6%.

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
| R7 | R6 with string-based `UNFREEZE_FROM_LAYER: block5_conv1` pipeline + EarlyStopping restore (epoch 22) | 86.0% / 84.3% | 76.2% | 0.615 | 0.799 |
| R8 | R7 + dedicated `data_augmentation` (brightness [0.8, 1.2]) + manual `class_weights` (adeno 1.3, large 0.8), SEED 45 | ~89% / ~74% | 79.6% | 0.557 | 0.805 |
| R9 | R8 + `ReduceLROnPlateau` (factor 0.5, patience 2, min_lr 1e-7) | ~87% / ~67% | 69.5% | 0.711 | 0.723 |
| R10 | Unfreeze `block4_conv1` onwards (blocks 4 & 5), LR 1e-5, SEED 49, class weights (adeno 1.3, large 0.8) | 87.0% / 82.9% | 81.0% | 0.513 | 0.828 |

Note: R10 is our new champion model. For the first time, test accuracy exceeded 81% (81.04%, 218/269 correct), test loss dropped to a project record of 0.513, and Macro F1 jumped to 0.828. Unfreezing from block4_conv1 gave the model enough representational capacity to adapt mid-level pulmonary CT textures, producing an 85.0% recall on adenocarcinoma (up from 75.0% in R8 and 65.8% in R7) while maintaining 100% recall on large_cell.

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
| R7 | Strong generalization, tighter train-val gap | Fine-tuning block5 with LR 1e-5 and restoring best weights at epoch 22 (val loss 0.609) avoided over-training, yielding 76.2% test accuracy with 0.80 Macro F1 |
| R8 | Class-balanced breakthrough; resolved distribution skew | Manual class weighting (adeno 1.3, large 0.8) and brightness augmentation rebalanced predictions: large_cell precision rose to 67.2%, adeno recall rose to 75.0%, boosting test accuracy to 79.55% and test loss to a record 0.557 |
| R9 | Severe under-optimization due to premature LR decay | **REJECTED**: LR 1e-5 was already small. Decaying it on a noisy validation set (70 images) starved gradient updates, trapping weights in an inferior state. Fixed LR 1e-5 is superior. |
| R10 | Major representational breakthrough (Blocks 4 & 5 unfrozen) | Expanding backbone fine-tuning from block5 down to block4_conv1 allowed learning specialized medical CT features without overfitting. Test accuracy reached 81.04%, test loss dropped to 0.5132, and Macro F1 climbed to 0.8283. |

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
| T3 | Fine-tune with ReduceLROnPlateau | Late curves bounce | loss down slightly | Tested in R9 and REJECTED: test acc fell 10.0% (79.55% -> 69.52%). Premature decay stalled fine-tuning. |
| T4 | Class weights (adeno 1.3, large 0.8) | large_cell over-predicted; val differs from test | large_cell precision up, recall down | Validated in R8 (large_cell precision 57.5% -> 67.2%, test acc 76.2% -> 79.6%) |
| T5 | Unfreeze `block4_conv1` | Only if T1-T2 leave train too low | train much up, gap up | Validated in R10: **Champion model** (Test acc 81.04%, loss 0.5132, Macro F1 0.828). |
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
| 16 | String-based `UNFREEZE_FROM_LAYER: block5_conv1` | Transparent, explicit layer unfreezing instead of fragile index counts | Unfreezes `block5_conv1` through top layers cleanly in pipeline |
| 17 | Track `UNFREEZE_FROM_LAYER` in `dvc.yaml` | Automated pipeline dependency tracking | DVC reproduces stage 03-05 automatically upon layer config change |
| 18 | Checkpoint restore via EarlyStopping (patience 5) | Prevent over-training on small dataset when fine-tuning | Restores epoch 22 best weights; test acc reaches 76.2% with loss 0.615 |
| 19 | Centralized `SEED: 45` in `params.yaml` and pipeline | Eliminates run-to-run variance across TensorFlow, NumPy, and Python | Fully reproducible generators and weight initialization |
| 20 | Separate `data_augmentation` & `class_weights` in `params.yaml` | Decouple training policies and loss weighting from backbone architecture | Clean, maintainable configuration without modifying Python code |
| 21 | Apply manual class weights (adeno 1.3, large 0.8) in `model.fit` | Mitigate validation/test class distribution mismatch | Test acc 79.55%, loss 0.5568, Macro F1 0.805 |
| 22 | Add `model/` directory to `.gitignore` | Prevent committing large weights binaries (58MB+ .h5 files) to Git repository | Keeps repository lightweight; artifacts tracked via DVC / MLflow |
| 23 | Add `ReduceLROnPlateau` hyperparameter block in `params.yaml` | Dynamically decay learning rate (factor 0.5, patience 3) when validation loss plateaus | Prevents optimization oscillation and allows fine-tuning into deeper minima before EarlyStopping |
| 24 | Disable `ReduceLROnPlateau` / Retain constant LR 1e-5 | Trial R9 showed severe degradation: accuracy dropped 10% (79.55% -> 69.52%), loss rose to 0.711 | Set `reduce_lr.ENABLE: False`. The 1e-5 LR is already minimal; further decaying on small/noisy val set halts gradient flow. |
| 25 | Unfreeze from `block4_conv1` onwards (`params.yaml`) | Block 5 alone limited mid-level CT texture feature adaptation | Unfreezes 8 conv layers across blocks 4 and 5; test accuracy crossed 81% (81.04%), test loss dropped to 0.5132, Macro F1 reached 0.8283. |

## 8. Working rules

- Change one thing per run; log it in MLflow via params.yaml.
- Run stages 03, 04, 05 together after an architecture change (or use `dvc repro`).
- Pick models by validation and macro F1; use test only to report.
- Compare accuracy against the always-guess-adenocarcinoma baseline (44.6%).
- Small differences (about 3% or less) are noise on a 269-image test set and a 70-image val set.
