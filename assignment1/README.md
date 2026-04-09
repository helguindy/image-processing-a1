# DMET 1001 — Image Processing Assignment 1

## Team Split

| Subteam | Members | Domain files |
|---|---|---|
| **A** | Habiba, Fareeda | `subteamA/domain_spatial.py`, `subteamA/domain_fourier.py` |
| **B** | Nour, Haya | `subteamB/domain_wavelet.py`, `subteamB/domain_custom.py` |

---

## Dataset

**CIFAR-10** — downloaded automatically the first time you run any training script. You do not need to download it manually.

- 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 60 000 images total (50 k train / 10 k test)
- Split applied in `dataset.py`: **70% train / 15% val / 15% test**
- The split is fixed by `SEED = 42` in `config.py` — all four domains see the exact same images

---

## Unified Parameters

These are locked in `config.py`. **Do not change them in your domain file — always import from config.**

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

## Shared Files 

| File | What it does |
|---|---|
| `config.py` | All unified parameters — **read-only for domain scripts** |
| `dataset.py` | Downloads CIFAR-10, applies the fixed split, returns DataLoaders |
| `model.py` | `get_model(in_channels)` — returns a MobileNetV2 with the right input/output |
| `evaluate.py` | Saves loss curves, confusion matrix, and metrics for any domain |

---

## How Each Domain Script Should Work

Every domain file follows the same structure:

1. **Define a transform function** — takes a `(3, 32, 32)` tensor, returns a `(C, 224, 224)` tensor
2. **Get dataloaders** — call `dataset.get_dataloaders(your_transform)` 
3. **Get model** — call `model.get_model(in_channels=C)`
4. **Training loop** — iterate for `config.NUM_EPOCHS`, using `config.LEARNING_RATE`, `config.BATCH_SIZE`, etc.
5. **Save results** — call `evaluate.save_results(...)` to output curves + confusion matrix

---

## Subteam A — Instructions

### domain_spatial.py
- Transform: resize `32×32 → 224×224` (bilinear), normalise with ImageNet mean/std
- `in_channels = 3`
- Standard baseline — expected to perform the best

### domain_fourier.py
- Apply `torch.fft.fft2` per channel on the `32×32` image
- Take the **log-magnitude**: `log(1 + |FFT|)`
- Resize result to `224×224`, normalise
- `in_channels = 3`
- In your report section: explain what the frequency domain captures and why it may or may not help classification

---

## Subteam B — Instructions

### domain_wavelet.py
- Use `PyWavelets` (`pip install PyWavelets`) — `pywt.dwt2(channel, 'haar')`
- One level of DWT gives 4 subbands per channel: LL, LH, HL, HH
- Option A: use only the LL (approximation) subband → `in_channels = 3`
- Option B: concatenate all 4 subbands → `in_channels = 12`
- Resize to `224×224`, normalise
- In your report section: justify your wavelet choice and subband selection

### domain_custom.py
- **Subteam B picks the domain** — suggestions: Sobel edges, Laplacian, HSV, Gabor filters
- Must justify the choice in the report
- Follow the same transform → dataloader → train → evaluate pattern

---

## Running

```bash
# install dependencies first
pip install -r requirements.txt

# run each domain script from the assignment1/ folder
python subteamA/domain_spatial.py
python subteamA/domain_fourier.py
python subteamB/domain_wavelet.py
python subteamB/domain_custom.py
```

Results (loss curves, confusion matrix, metrics) are saved automatically under a `results/` folder.

---

## Deliverables per Domain

- [ ] Training accuracy curve
- [ ] Validation accuracy curve
- [ ] Test accuracy
- [ ] Loss curves
- [ ] Confusion matrix
