import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, RobertaConfig, RobertaModel
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import argparse
import random
import os

# ------------------------
# LOAD DATA
# ------------------------
def load_data(dataset_path, random_state, min_user_comments):
    df = pd.read_csv(dataset_path)
    df["text"] = df["text"].fillna("").astype(str)
    df["user_id"] = df["user_id"].astype(int)

    # 1. CONTEGGIO COMMENTI E FILTRO SOGLIA
    if min_user_comments > 0:
        # Conta quanti commenti ha ogni utente
        user_counts = df["user_id"].value_counts()
        # Identifica gli utenti che non raggiungono la soglia
        users_to_mask = user_counts[user_counts < min_user_comments].index
        # Imposta a 0 l'ID di quegli utenti
        df.loc[df["user_id"].isin(users_to_mask), "user_id"] = 0

    # 2. MAPPING (Solo per gli utenti rimasti sopra soglia + l'ID 0)
    # Prendiamo gli ID unici rimasti (se min_comments > 0, molti saranno già 0)
    user_ids_unique = [uid for uid in df["user_id"].unique() if uid != 0]
    
    # Creiamo il mapping: ID reali partono da 1
    user_to_idx = {uid: idx + 1 for idx, uid in enumerate(user_ids_unique)}
    user_to_idx[0] = 0  # Lo 0 punta all'indice 0 (padding/unknown)
    
    # Applichiamo il mapping
    df["user_idx"] = df["user_id"].apply(lambda x: user_to_idx.get(x, 0))
    num_users = len(user_to_idx)

    # 3. SPLIT
    # Nota: se molti utenti sono diventati '0', la stratificazione è ancora possibile
    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=random_state,
        stratify=df["user_id"].astype(str)
    )

    return train_df, val_df, user_to_idx, num_users


def augment_word_lists(df, n_aug=1):
    mask = df["is_words"] == 1
    words_df = df[mask].copy()
    
    aug_rows = []
    
    for _, row in words_df.iterrows():
        # Puliamo e normalizziamo i termini
        original_words = [w.strip() for w in row["text"].split(",")]
        # Usiamo un set per memorizzare le stringhe già create (inclusa l'originale)
        seen_combinations = {", ".join(original_words)}
        
        # Calcoliamo il numero massimo di permutazioni possibili (n!) 
        # per non entrare in un loop infinito se n_aug è troppo alto
        import math
        max_possible = math.factorial(len(original_words))
        # Quante ne vogliamo effettivamente (originale + n_aug)
        target_count = min(max_possible, n_aug + 1)
        
        attempts = 0
        # Continuiamo finché non raggiungiamo il target o finiamo i tentativi (max 100)
        while len(seen_combinations) < target_count and attempts < 100:
            shuffled = random.sample(original_words, len(original_words))
            combined = ", ".join(shuffled)
            
            if combined not in seen_combinations:
                seen_combinations.add(combined)
                new_row = row.copy()
                new_row["text"] = combined
                aug_rows.append(new_row)
            
            attempts += 1
            
    if aug_rows:
        aug_df = pd.DataFrame(aug_rows)
        df = pd.concat([df, aug_df], ignore_index=True)
        # Mischiamo il dataset finale per non avere tutti gli aumentati in coda
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


# ------------------------
# TARGET NORMALIZATION (TRAIN ONLY)
# ------------------------
def normalize_data(train_df, val_df):
    y_train = train_df[["valence", "arousal"]].values.astype(np.float32)
    mu = y_train.mean(axis=0)
    sigma = y_train.std(axis=0) + 1e-8

    train_df[["valence", "arousal"]] = (y_train - mu) / sigma
    val_df[["valence", "arousal"]] = (val_df[["valence", "arousal"]].values - mu) / sigma

    return train_df, val_df, mu, sigma


# ------------------------
# DATASET
# ------------------------
class AffectDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor([row["valence"], row["arousal"]], dtype=torch.float),
            "user_idx": torch.tensor(row["user_idx"], dtype=torch.long),
            "user_id": torch.tensor(row["user_id"], dtype=torch.long)
        }


class AffectTestDataset(Dataset):
    def __init__(self, df, user_to_idx, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.user_to_idx = user_to_idx
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        u_idx = self.user_to_idx.get(row["user_id"], 0)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "user_idx": torch.tensor(u_idx, dtype=torch.long),
            "user_id": row["user_id"],
            "text_id": row["text_id"]
        }
    

# ------------------------
# MODEL: USER EMBEDDING
# ------------------------
class AffectModelWithUserEmb(nn.Module):
    def __init__(self, encoder_model_name, block_encoder_weights, hidden_dim_mlp, dropout_value, user_emb_dim, number_of_user):
        super().__init__()
        self.encoder_model_name = encoder_model_name
        self.block_encoder_weights = block_encoder_weights
        self.hidden_dim_mlp = hidden_dim_mlp
        self.dropout_value = dropout_value
        self.user_emb_dim = user_emb_dim
        self.number_of_user = number_of_user
        
        if self.encoder_model_name == "":
            config = RobertaConfig.from_pretrained("roberta-base")
            model = RobertaModel(config)
            self.roberta = model
            self.block_encoder_weights = False
        else:
            self.roberta = AutoModel.from_pretrained(self.encoder_model_name)

        if self.block_encoder_weights == True:
            for param in self.roberta.parameters():
                param.requires_grad = False

        if self.user_emb_dim >0:
            self.user_emb = nn.Embedding(self.number_of_user, self.user_emb_dim, padding_idx=0)
            nn.init.normal_(self.user_emb.weight, mean=0, std=0.02)

        self.regressor = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size + self.user_emb_dim, self.hidden_dim_mlp),
            nn.ReLU(),
            nn.Dropout(self.dropout_value),
            nn.Linear(self.hidden_dim_mlp, 2)
        )

    def forward(self, input_ids, attention_mask, user_idx):
        text_emb = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]  # CLS

        if self.user_emb_dim > 0:
            user_emb = self.user_emb(user_idx)
            x = torch.cat([text_emb, user_emb], dim=1)
        else:
            x = text_emb

        return self.regressor(x)
    

# ------------------------
# METRICS
# ------------------------
def compute_between_r(y_true, y_pred, user_ids):
    yt, yp = [], []
    for u in np.unique(user_ids):
        m = user_ids == u
        yt.append(y_true[m].mean())
        yp.append(y_pred[m].mean())
    r, _ = pearsonr(yt, yp)
    return r if not np.isnan(r) else 0.0

def compute_within_r(y_true, y_pred, user_ids):
    rs = []
    for u in np.unique(user_ids):
        m = user_ids == u
        if m.sum() > 1:
            r, _ = pearsonr(y_true[m], y_pred[m])
            if not np.isnan(r):
                rs.append(r)
    return np.mean(rs) if rs else 0.0

def composite_r(rw, rb):
    return np.tanh((np.arctanh(rw) + np.arctanh(rb)) / 2)

# ------------------------
# CREATE MODEL AND TOKENIZER
# ------------------------
def create_model_and_tokenizer(encoder_model_name, block_encoder_weights, hidden_dim_mlp, dropout_value, user_emb_dim, number_of_user):
    if encoder_model_name == "":
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    else:
        tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)

    model = AffectModelWithUserEmb(encoder_model_name= encoder_model_name,
                                   block_encoder_weights= block_encoder_weights,
                                   hidden_dim_mlp= hidden_dim_mlp,
                                   dropout_value= dropout_value,
                                   user_emb_dim= user_emb_dim,
                                   number_of_user= number_of_user,
                                   )
    return model, tokenizer

# ------------------------
# TRAINING
# ------------------------
def traing(device, train_df, val_df, batch_size, learning_rate, epochs, sigma, mu, save_model_path, tokenizer, max_len, model):
    
    model.to(device)

    train_loader = DataLoader(AffectDataset(train_df, tokenizer=tokenizer, max_len=max_len), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(AffectDataset(val_df, tokenizer=tokenizer, max_len=max_len), batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.SmoothL1Loss(beta=0.5)  # HUBER

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri allenabili: {trainable_params}")

    best_score = -1
    best_epoch = -1

    for epoch in range(1, epochs + 1):

        model.train()
        total_loss = 0.0

        for b in train_loader:
            optimizer.zero_grad()
            preds = model(
                b["input_ids"].to(device),
                b["attention_mask"].to(device),
                b["user_idx"].to(device)
            )
            loss = criterion(preds, b["labels"].to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # ---- VALIDATION ----
        model.eval()
        P, Y, U = [], [], []

        with torch.no_grad():
            for b in val_loader:
                p = model(
                    b["input_ids"].to(device),
                    b["attention_mask"].to(device),
                    b["user_idx"].to(device)
                ).cpu().numpy()
                P.append(p)
                Y.append(b["labels"].numpy())
                U.append(b["user_id"].numpy())

        P = np.vstack(P) * sigma + mu
        Y = np.vstack(Y) * sigma + mu
        U = np.concatenate(U)

        rv = composite_r(
            compute_within_r(Y[:, 0], P[:, 0], U),
            compute_between_r(Y[:, 0], P[:, 0], U)
        )
        ra = composite_r(
            compute_within_r(Y[:, 1], P[:, 1], U),
            compute_between_r(Y[:, 1], P[:, 1], U)
        )

        avg = (rv + ra) / 2
        print(f"Epoch {epoch} | Loss {total_loss/len(train_loader):.4f} | Val {rv:.3f} | Aro {ra:.3f} | Avg {avg:.3f}")

        if avg > best_score:
            best_score = avg
            best_val = rv
            best_aro = ra
            best_epoch = epoch
            torch.save(model.state_dict(), save_model_path)
            print(" → Saved best model")

    print(f"Best model at epoch: {best_epoch:4d}  -> avg r_composite: {best_score:6.4f} with Valence: {best_val:6.4f} and Arousal: {best_aro:6.4f}")
    return best_epoch, best_val, best_aro, best_score


# ------------------------
# FINAL INFERENCE
# ------------------------
def testing(device, model_path, test_ds_path, user_to_idx, batch_size, sigma, mu, tokenizer, max_len, model):
    model.to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    test_df = pd.read_csv(test_ds_path)
    test_df["text"] = test_df["text"].fillna("").astype(str)

    test_loader = DataLoader(
        AffectTestDataset(test_df, user_to_idx, tokenizer=tokenizer, max_len=max_len),
        batch_size=batch_size
    )

    rows = []

    with torch.no_grad():
        for b in test_loader:
            preds = model(
                b["input_ids"].to(device),
                b["attention_mask"].to(device),
                b["user_idx"].to(device)
            ).cpu().numpy()

            preds = preds * sigma + mu

            for i in range(len(preds)):
                rows.append({
                    "user_id": b["user_id"][i].item(),
                    "text_id": b["text_id"][i].item(),
                    "pred_valence": preds[i, 0],
                    "pred_arousal": preds[i, 1]
                })

    pd.DataFrame(rows).to_csv("pred_subtask1.csv", index=False)
    print("Submission saved.")


def argument():
    # "roberta-base", "cardiffnlp/twitter-roberta-base-emotion, SamLowe/roberta-base-go_emotions
    parser = argparse.ArgumentParser(description="Affect Model Training Configuration")

    parser.add_argument("--model_name", type=str, default="roberta-base", help="HuggingFace model name")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--user_emb_dim", type=int, default=32, help="User embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze roberta weights")
    parser.add_argument("--save_path", type=str, default="best_model_task1_user_emb.pth", help="Model save path")
    parser.add_argument("--train_path", type=str, default="data/train_subtask1.csv", help="Train data path")
    parser.add_argument("--test_path", type=str, default="data/test_subtask1.csv", help="Test data path")
    parser.add_argument("--random_state", type=int, default=0, help="Random seed")
    parser.add_argument("--min_user_comments", type=int, default=0, help="Set id to 0 from the user that have less than N comments")
    parser.add_argument("--augment_dataset_value", type=int, default=0, help="Number of ricombination of phrase for text that are sequence of words")

    return parser.parse_args()


def write_training_report(
    fine_report_name,
    model_name, max_len, batch_size, learing_rate, hidden_dim, dropout,
    user_embedding_dim, freeze_encoder,
    filter_few_comments_user, augment_dataset,
    random_state, best_epoch,
    best_valence_value, best_arousal_value, best_mean_value
):
    file_exists = os.path.isfile(fine_report_name)

    headers = [
        "model_name", "max_len", "batch_size", "learing_rate", "hidden_dim",
        "dropout", "user_embedding_dim", "freeze_encoder",
        "filter_few_comments_user", "augment_dataset",
        "random_state", "best_epoch",
        "best_valence_value", "best_arousal_value", "best_mean_value"
    ]

    values = [
        model_name, max_len, batch_size, learing_rate, hidden_dim,
        dropout, user_embedding_dim, freeze_encoder,
        filter_few_comments_user, augment_dataset,
        random_state, best_epoch,
        best_valence_value, best_arousal_value, best_mean_value
    ]

    with open(fine_report_name, "a", encoding="utf-8") as f:
        # Se il file non esiste, scrive l'header
        if not file_exists:
            f.write(",".join(headers) + "\n")

        # Scrive i valori
        f.write(",".join(map(str, values)) + "\n")

def main():

    args = argument()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_NAME = args.model_name
    MAX_LEN = args.max_len
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs

    USER_EMB_DIM = args.user_emb_dim
    HIDDEN_DIM = args.hidden_dim
    DROPOUT = args.dropout
    FREEZE_ENCODER = args.freeze_encoder

    SAVE_BEST_MODEL_PATH = args.save_path
    TRAIN_PATH = args.train_path
    TEST_PATH = args.test_path

    RANDOM_STATE = args.random_state
    N_COMMENTS_TO_BE_KNOW = args.min_user_comments
    AUGMENT_DATASET_VALUE = args.augment_dataset_value

    torch.manual_seed(RANDOM_STATE)
    train_df, val_df, user_to_idx, num_users = load_data(TRAIN_PATH, RANDOM_STATE, N_COMMENTS_TO_BE_KNOW)
    train_df, val_df, mu, sigma = normalize_data(train_df, val_df)
    print(f"Number of elments in training dataset: {len(train_df)}")
    if AUGMENT_DATASET_VALUE > 0:
        train_df = augment_word_lists(train_df, AUGMENT_DATASET_VALUE)
        print(f"Number of elments after augmentation in training dataset: {len(train_df)}")
    model, tokenizer = create_model_and_tokenizer(MODEL_NAME, FREEZE_ENCODER, HIDDEN_DIM, DROPOUT, USER_EMB_DIM, num_users)
    best_epoch, best_valence, best_arousal, best_score = traing(DEVICE, train_df, val_df, BATCH_SIZE, LR, EPOCHS, sigma, mu, SAVE_BEST_MODEL_PATH, tokenizer, MAX_LEN, model)
    write_training_report("task1/report/report_training_file.csv",
                          MODEL_NAME, MAX_LEN, BATCH_SIZE, LR, HIDDEN_DIM, DROPOUT, USER_EMB_DIM, FREEZE_ENCODER, N_COMMENTS_TO_BE_KNOW, AUGMENT_DATASET_VALUE, RANDOM_STATE,
                          best_epoch, best_valence, best_arousal, best_score)
    testing(DEVICE, SAVE_BEST_MODEL_PATH, TEST_PATH, user_to_idx, BATCH_SIZE, sigma, mu, tokenizer, MAX_LEN, model)


if __name__ == "__main__":
    main()