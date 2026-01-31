# EmotionValArouTimeVariation2026 — Task 2 (Subtask 2A / 2B)

A single Python runner that executes **Subtask 2A** or **Subtask 2B** based on a CLI flag:

- `-a` → runs **Subtask 2A** (state change forecasting)
- `-b` → runs **Subtask 2B** (dispositional change forecasting)

Main script: `task2/task2.py` (or `run_subtask2.py` if you renamed it).

---

## Recommended project structure

```
EmotionValArouTimeVariation2026/
├── datasets/
│   ├── train_subtask2a.csv
│   ├── subtask2a_forecasting_user_marker.csv
│   ├── train_subtask2b_detailed.csv
│   └── subtask2b_forecasting_user_marker.csv
├── task2/
│   └── task2.py
├── eval.py                  # optional: provides task2_correlation
├── requirements.txt
└── README.md
```

> The script uses `pathlib` to resolve dataset paths robustly:
> - `BASE_DIR = Path(__file__).resolve().parent.parent`
> - `DATA_DIR = BASE_DIR / "datasets"`

---

## Installation

### 1) Create and activate a virtual environment (recommended)

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running

### Subtask 2A
```bash
python task2/task2.py -a
```

Change output:
```bash
python task2/task2.py -a --out_path_2a my_2a.csv
```

### Subtask 2B
```bash
python task2/task2.py -b
```

Change output:
```bash
python task2/task2.py -b --out_path_2b my_2b.csv
```

---

## Key CLI arguments

### Shared
```bash
--seed
```

### Subtask 2A
```bash
--model_name_2a
--max_len_2a
--batch_size_2a
--lr_2a
--epochs_2a
--k_history_2a
```

### Subtask 2B
```bash
--model_name_2b
--k_texts_2b
--text_sampling_2b
--epochs_2b
--patience_2b
--n_splits_2b
```

---

## Output format

### 2A
- user_id  
- state_change_valence  
- state_change_arousal  

### 2B
- user_id  
- disposition_change_valence  
- disposition_change_arousal  

---

## Optional evaluation

If `eval.py` is present and exposes `task2_correlation`, the script computes unofficial scores.

---

## Troubleshooting

### File not found
```bash
ls datasets
```

### Slow on CPU (2B)
```bash
python task2/task2.py -b --epochs_2b 10 --n_splits_2b 3
```

---

## Leakage note (2B)

2B uses only **group==1** information and safe features.

---

## License
See LICENSE if provided.
