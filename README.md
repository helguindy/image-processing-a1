## Execution Steps

**1. Download the submission folder from Google Drive and unzip it.**

The folder contains:
```text
assignment1/
├── notebook.ipynb
├── requirements.txt
├── README.md
└── results/
    ├── spatial_weights.pth
    ├── fourier_weights.pth
    ├── wavelet_weights.pth
    └── custom_weights.pth
```

**2. Create and activate a virtual environment:**
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Open the notebook:**

Open `notebook.ipynb` directly in VS Code or run:
```bash
jupyter notebook notebook.ipynb
```

**5. Run all cells top to bottom in order:**

| Cell | Description |
|---|---|
| Cell 1 | Imports all libraries |
| Cell 2 | Sets all shared hyperparameters |
| Cell 3 | Defines the dataset loader |
| Cell 4 | Defines the MobileNetV2 model |
| Cell 5 | Defines training and evaluation utilities |
| Domain 1 cells | Spatial transform → visualise → train → results |
| Domain 2 cells | Fourier transform → visualise → train → results |
| Domain 3 cells | Wavelet transform → visualise → train → results |
| Domain 4 cells | HSV transform → visualise → train → results |
| Final comparison | Prints accuracy table and plots all validation curves |
| Save curves cell | Saves all result plots to `results/` folder |

> **Important:** Restart the kernel before running. Always run all cells top to bottom in a single clean run.

**6. Notes:**
- CIFAR-10 downloads automatically to a `data/` folder — no manual dataset setup needed.
- Training runs on CPU. Expect 15–20 minutes per domain on a standard laptop.
- All output files are saved automatically to the `results/` folder.
