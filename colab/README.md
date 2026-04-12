# 🚀 Google Colab Notebooks for LoRA Compress

Run autoresearch and model compression on free GPU resources (T4) with results stored in Google Drive.

## 📓 Notebooks

### 1. `LoRA_Compress_Autoresearch.ipynb`
Hyperparameter search to find optimal LoRA configs for a single layer.

**Use when:**
- You want to find the best rank/LR/epochs for your quality target
- Your local CPU is too slow (357s/trial → ~10s/trial on T4!)

**Time:** ~15-30 min for 50 trials

**Output:** 
- Optuna database in Drive (`databases/optuna_l1_quality_X.Xpct.db`)
- JSON results with best configs

### 2. `LoRA_Compress_Full_Model.ipynb`
Compress an entire model layer-by-layer with Drive storage.

**Use when:**
- You want to compress the full model
- You need resume capability (Colab may disconnect)

**Time:** ~1-3 hours for full model

**Output:**
- Compressed layers in `compressed_models/full_model/layers/`
- Metadata JSON with quality metrics

## 🚀 Quick Start

1. **Open notebook in Colab:**
   - File → Upload notebook
   - Or open from GitHub: `https://github.com/qades/loracompress/tree/main/colab`

2. **Set runtime to GPU:**
   - Runtime → Change runtime type → T4 GPU

3. **Run cells in order**

4. **Authorize Google Drive** when prompted

## 📁 Drive Structure

```
MyDrive/LoRA_Compress/
├── databases/           # Optuna studies (persistent)
│   ├── optuna_l1_quality_3.0pct.db
│   └── optuna_l1_quality_5.0pct.db
├── compressed_models/   # Full model compression
│   └── full_model/
│       ├── metadata.json
│       └── layers/
├── results/            # JSON outputs
└── checkpoints/        # Training checkpoints
```

## ⚡ Speed Comparison

| Platform | Per Trial | 50 Trials |
|----------|-----------|-----------|
| Strix Halo CPU | ~360s | ~5 hours |
| Colab CPU | ~100s | ~1.5 hours |
| **Colab T4 GPU** | **~5-15s** | **~10-20 min** |

**~20-30× speedup vs your local CPU!**

## 🔄 Resume Capability

Both notebooks support resuming:
- Databases are stored in Drive
- Compressed layers are checked before re-processing
- Just re-run the notebook if disconnected

## ⚠️ Colab Limitations

- **Session timeout:** ~12 hours max (90 min idle)
- **GPU not guaranteed:** May get CPU fallback
- **State loss:** Variables lost on disconnect (but Drive files persist)

## 📊 Interpreting Results

### Autoresearch Scores
- **Negative scores (e.g., -150):** ✅ Hit target with good compression
- **0-1000:** ⚠️ Barely hit target
- **1000+:** ❌ Missing target (penalty applied)

### Quality Targets
- **3% L1 error:** Q4_K_M benchmark quality (strict)
- **5% L1 error:** Good balance of speed/quality (easier)

## 🛠️ Troubleshooting

**"CUDA out of memory"**
- Restart runtime and try again
- Reduce batch size if training

**"GPU not available"**
- Runtime → Change runtime type → Confirm T4 is selected
- Try again later if GPUs are at capacity

**"Drive not mounting"**
- Re-run the mount cell
- Check Google account permissions

## 🔗 Links

- [GitHub Repository](https://github.com/qades/loracompress)
- [Open Notebook 1 (Autoresearch)](https://colab.research.google.com/github/qades/loracompress/blob/main/colab/LoRA_Compress_Autoresearch.ipynb)
- [Open Notebook 2 (Full Model)](https://colab.research.google.com/github/qades/loracompress/blob/main/colab/LoRA_Compress_Full_Model.ipynb)
