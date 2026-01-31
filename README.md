# EmotionValArouTimeVariation2026 — Task 2 (Subtask 1)

For running the task1 is needed to install the following libraries:
- pandas
- numpy
- torch 
- transformer
- scikit-learn
- scipy

```
pip install pandas numpy torch transformers scikit-learn scipy
```

The script **`task1/task1.py`** supports the following command-line arguments to customize the training and inference process:

* **Model & Architecture Configuration**
    * `--model_name`: The HuggingFace pre-trained model to use (e.g., `roberta-base`).
    * `--user_emb_dim`: Dimensionality of the user embedding vector. Set to 0 to disable user-specific modeling.
    * `--hidden_dim`: Dimensionality of the hidden layer in the final MLP regressor.
    * `--dropout`: Dropout rate applied to the MLP to prevent overfitting.
    * `--max_len`: Maximum sequence length for tokenization.
    * `--freeze_encoder`: If provided, freezes the weights of the transformer encoder, training only the regression head.

* **Training Hyperparameters**
    * `--batch_size`: Number of samples per training batch.
    * `--lr`: Learning rate for the AdamW optimizer.
    * `--epochs`: Total number of training epochs.
    * `--random_state`: Random seed for reproducibility of splits and weight initialization.

* **Data Paths & Processing**
    * `--train_path`: Path to the CSV file containing the training data.
    * `--test_path`: Path to the CSV file containing the test data for final inference.
    * `--save_path`: Destination path to save the best model weights (`.pth`).
    * `--min_user_comments`: Minimum comment threshold for users; those with fewer comments are mapped to a generic ID (0).
    * `--augment_dataset_value`: Number of random recombinations to generate for samples identified as word lists (`is_words == 1`).
    * `--report_path_save`: Destination path where the training report and its parameters will be saved as a CSV.

The script also save the results and parameter of the training in a csv file. 

The script **`task1/run_task1_many_times.py`** is a script that allow to run more times in a unique script **task1.py**, it is usefull for recreate a grid search

The script **`task1/read_results.py`** is a script for get a quickly overview of a csv of different training, return the results in a txt file.

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
│   ├──subtask2b_forecasting_user_marker.csv
│   └── train_subtask1.csv
├── task1/
│   └── task1.py
├── task2/
│   └── task2.py
├── eval.py                 
├── requirements.txt
└── README.md
```

> The script uses `pathlib` to resolve dataset paths robustly:
> - `BASE_DIR = Path(__file__).resolve().parent.parent`
> - `DATA_DIR = BASE_DIR / "datasets"`

---

## Installation

### Install dependencies

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

