# DMET 1001 — Image Processing Assignment 1
**German University in Cairo** | Dr. Mohamed Karam Gabr | **Due: 14 April 2026**

---

## Team Split

| Subteam | Members | Domains |
|---|---|---|
| **A** | Habiba, Fareeda | Spatial, Fourier |
| **B** | Nour, Haya | Wavelet, Custom |

---

## Project Structure

```
assignment1/
├── notebook.ipynb    ← everything is here — open and run this
├── requirements.txt  ← dependencies
├── data/             ← CIFAR-10 downloads here automatically
└── results/          ← model weights + plots saved here after training
```

---

## Dataset

**CIFAR-10** (5 classes) — downloads automatically the first time you run the notebook.

- Using 5 out of 10 classes: **airplane, automobile, bird, ship, truck**
- ~30 000 training images (5 classes × 5 000 each) | Split: **70% train / 15% val / 15% test**
- Split is fixed with `SEED = 42` — all 4 domains see the exact same images
- To change the classes, edit `SELECTED_CLASSES` in Cell 2 of the notebook

---

## Unified Parameters

All parameters are defined in **Cell 2** of the notebook. Do not change them.

| Parameter | Value |
|---|---|
| Model | MobileNetV2 (PyTorch, pretrained on ImageNet) |
| Input size | 224 × 224 |
| Batch size | 64 |
| Optimizer | Adam |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Epochs | 20 |
| Random seed | 42 |
| Split | 70 / 15 / 15 |

---

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Open the notebook
jupyter notebook notebook.ipynb

# 3. Run all cells top to bottom (Kernel → Restart & Run All)
```

---

## Notebook Structure

| Cell | What it contains |
|---|---|
| 1 | Imports |
| 2 | Unified parameters (config) |
| 3 | Dataset loader — shared |
| 4 | Model — shared |
| 5 | Training & evaluation utilities — shared |
| 6–9 | **Domain 1: Spatial** (Subteam A) |
| 10–13 | **Domain 2: Fourier** (Subteam A) |
| 14–17 | **Domain 3: Wavelet** (Subteam B) |
| 18–21 | **Domain 4: Custom** (Subteam B) |
| 22–23 | Final comparison — all 4 domains |

---

## What Each Subteam Implements

Each domain section has:
1. The **transform function** — this is what each subteam writes
2. Visualisation of a few transformed images
3. A `train_model(...)` call — already written, just run it
4. A `plot_results(...)` call — already written, just run it

**Subteam A** fills in `spatial_transform` and `fourier_transform`  
**Subteam B** fills in `wavelet_transform` and `custom_transform`

---

## Deliverables Checklist

### Per Domain (4 total)
- [ ] Training accuracy curve
- [ ] Validation accuracy curve
- [ ] Test accuracy
- [ ] Loss curves
- [ ] Confusion matrix
- [ ] Per-class accuracy *(optional bonus)*

### Final Submission (upload everything to Google Drive, share link)
- [ ] `notebook.ipynb` — all cells run with visible outputs
- [ ] Report (PDF)
- [ ] Model weights — saved automatically to `results/` after training
- [ ] This README

### Submit via
**https://forms.gle/ghRoTCsokhMHmZf47**

---
