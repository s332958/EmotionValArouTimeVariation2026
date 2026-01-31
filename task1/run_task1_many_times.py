import os
import torch
import itertools


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ENCODER_NAME = ["cardiffnlp/twitter-roberta-base-emotion", "SamLowe/roberta-base-go_emotions", "roberta-base"]
MAX_LEN = [128]
BATCH_SIZE = [16]
LR = [2e-5]
EPOCHS = 10

USER_EMB_DIM = [0, 32]
HIDDEN_DIM = [128]
DROPOUT = [0.2]
FREEZE_ENCODER = False

SAVE_BEST_MODEL_PATH = "best_model.pth"
TRAIN_PATH = "datasets/train_subtask1.csv"

RANDOM_STATE = [0, 100, 200, 300]
N_COMMENTS_TO_BE_KNOW = [0, 10]
AUGMENT_DATASET_VALUE = [0, 3]

for encoder_name in ENCODER_NAME:
    for max_len in MAX_LEN:
        for batch_size in BATCH_SIZE:
            for lr in LR:
                for user_emb_dim in USER_EMB_DIM:
                    for hidden_dim in HIDDEN_DIM:
                        for dropout in DROPOUT:
                            for random_state in RANDOM_STATE:
                                for augment_dataset_value in AUGMENT_DATASET_VALUE:
                                    if user_emb_dim > 0:
                                        for n_comments_to_be_know in N_COMMENTS_TO_BE_KNOW:
                                            command_line = f"python task1/task1_v2.py --model_name {encoder_name} --max_len {max_len} --batch_size {batch_size} --lr {lr} --epochs {EPOCHS} --user_emb_dim {user_emb_dim} --hidden_dim {hidden_dim} --dropout {dropout} --random_state {random_state} --min_user_comments {n_comments_to_be_know} --augment_dataset_value {augment_dataset_value} --train_path {TRAIN_PATH}  --test_path {TRAIN_PATH}"
                                            print(command_line)         
                                            os.system(command_line)     
                                    else:
                                        n_comments_to_be_know = 0
                                        command_line = f"python task1/task1_v2.py --model_name {encoder_name} --max_len {max_len} --batch_size {batch_size} --lr {lr} --epochs {EPOCHS} --user_emb_dim {user_emb_dim} --hidden_dim {hidden_dim} --dropout {dropout} --random_state {random_state} --min_user_comments {n_comments_to_be_know} --augment_dataset_value {augment_dataset_value} --train_path {TRAIN_PATH}  --test_path {TRAIN_PATH}"
                                        print(command_line)          
                                        os.system(command_line)      