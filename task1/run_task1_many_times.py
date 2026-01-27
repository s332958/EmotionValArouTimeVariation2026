import os
import torch
import itertools


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENCODER_NAME = ["roberta-base", "cardiffnlp/twitter-roberta-base-emotion", "SamLowe/roberta-base-go_emotions"]
MAX_LEN = [128]
BATCH_SIZE = [16]
LR = [2e-5]
EPOCHS = 10

USER_EMB_DIM = [0, 16, 32]
HIDDEN_DIM = [128]
DROPOUT = [0.2]
FREEZE_ENCODER = [True, False]

SAVE_BEST_MODEL_PATH = "best_model.pth"
TRAIN_PATH = "datasets/train_subtask1.csv"
TEST_PATH = "" #"datasets/train_subtask1.csv"

RANDOM_STATE = [0, 100, 200, 300]
N_COMMENTS_TO_BE_KNOW = [0, 5, 10]
AUGMENT_DATASET_VALUE = [0, 3, 6]

for (
    encoder_name,
    max_len,
    batch_size,
    lr,
    user_emb_dim,
    hidden_dim,
    dropout,
    random_state,
    n_comments_to_be_know,
    augment_dataset_value
) in itertools.product(
    ENCODER_NAME,
    MAX_LEN,
    BATCH_SIZE,
    LR,
    USER_EMB_DIM,
    HIDDEN_DIM,
    DROPOUT,
    RANDOM_STATE,
    N_COMMENTS_TO_BE_KNOW,
    AUGMENT_DATASET_VALUE
):
    command_line = f"python task1/task1_v2.py --model_name {encoder_name} --max_len {max_len} --batch_size {batch_size} --lr {lr} --epochs {EPOCHS} --user_emb_dim {user_emb_dim} --hidden_dim {hidden_dim} --dropout {dropout} --random_state {random_state} --min_user_comments {n_comments_to_be_know} --augment_dataset_value {augment_dataset_value} --train_path {TRAIN_PATH}  --test_path {TRAIN_PATH}"
    print(command_line)          # 🔍 debug / log
    os.system(command_line)      # 🚀 esegue