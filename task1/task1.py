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

# ------------------------
# ARGPARSE SETUP
# ------------------------
parser = argparse.ArgumentParser(description="Affect Model Training Configuration")

parser.add_argument("--model_name", type=str, default="roberta-base", help="HuggingFace model name")
parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--user_emb_dim", type=int, default=32, help="User embedding dimension")
parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension")
parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
parser.add_argument("--freeze_roberta", action="store_true", help="Freeze roberta weights")
parser.add_argument("--save_path", type=str, default="best_model_task1_user_emb.pth", help="Model save path")
parser.add_argument("--train_path", type=str, default="data/train_subtask1.csv", help="Train data path")
parser.add_argument("--test_path", type=str, default="data/test_subtask1.csv", help="Test data path")
parser.add_argument("--random_state", type=int, default=0, help="Random seed")
parser.add_argument("--remove_id_less_user", type=int, default=0, help="Set id to 0 from the user that have less than N comments")

# "roberta-base", "cardiffnlp/twitter-roberta-base-emotion, SamLowe/roberta-base-go_emotions

args = parser.parse_args()

# ------------------------
# CONFIG
# ------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = args.model_name
MAX_LEN = args.max_len
BATCH_SIZE = args.batch_size
LR = args.lr
EPOCHS = args.epochs

USER_EMB_DIM = args.user_emb_dim
HIDDEN_DIM = args.hidden_dim
DROPOUT = args.dropout
FREEZE_ROBERTA = args.freeze_roberta

SAVE_PATH = args.save_path
TRAIN_PATH = args.train_path
TEST_PATH = args.test_path

RANDOM_STATE = args.random_state
N_COMMENTS = args.remove_id_less_user

# Importante per la riproducibilità
torch.manual_seed(RANDOM_STATE)

# ------------------------
# LOAD DATA
# ------------------------
df = pd.read_csv(TRAIN_PATH)
df["text"] = df["text"].fillna("").astype(str)
df["user_id"] = df["user_id"].astype(int)

# user_id → indice continuo
user_ids_unique = df["user_id"].unique()
user_to_idx = {uid: idx for idx, uid in enumerate(user_ids_unique)}
df["user_idx"] = df["user_id"].map(user_to_idx)
num_users = len(user_to_idx)

train_df, val_df = train_test_split(
    df,
    test_size=0.1,
    random_state=RANDOM_STATE,
    stratify=df["user_id"].astype(str)
)

# ------------------------
# TARGET NORMALIZATION (TRAIN ONLY)
# ------------------------
y_train = train_df[["valence", "arousal"]].values.astype(np.float32)
mu = y_train.mean(axis=0)
sigma = y_train.std(axis=0) + 1e-8

train_df[["valence", "arousal"]] = (y_train - mu) / sigma
val_df[["valence", "arousal"]] = (val_df[["valence", "arousal"]].values - mu) / sigma

# ------------------------
# TOKENIZER
# ------------------------
if MODEL_NAME == "":
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
else:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ------------------------
# DATASET
# ------------------------
class AffectDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor([row["valence"], row["arousal"]], dtype=torch.float),
            "user_idx": torch.tensor(row["user_idx"] + 1, dtype=torch.long),
            "user_id": torch.tensor(row["user_id"], dtype=torch.long)
        }

class AffectTestDataset(Dataset):
    def __init__(self, df, user_to_idx):
        self.df = df.reset_index(drop=True)
        self.user_to_idx = user_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        u_idx = self.user_to_idx.get(row["user_id"], -1) + 1
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
    def __init__(self):
        super().__init__()
        actual_freeze = FREEZE_ROBERTA
        
        if MODEL_NAME == "":
            config = RobertaConfig.from_pretrained("roberta-base")
            model = RobertaModel(config)
            self.roberta = model
            actual_freeze = False
        else:
            self.roberta = AutoModel.from_pretrained(MODEL_NAME)

        if actual_freeze == True:
            for param in self.roberta.parameters():
                param.requires_grad = False

        if USER_EMB_DIM >0:
            self.user_emb = nn.Embedding(num_users + 1, USER_EMB_DIM, padding_idx=0)
            nn.init.normal_(self.user_emb.weight, mean=0, std=1)

        self.regressor = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size + USER_EMB_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, 2)
        )

    def forward(self, input_ids, attention_mask, user_idx):
        text_emb = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]  # CLS

        if USER_EMB_DIM > 0:
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
# TRAINING
# ------------------------
model = AffectModelWithUserEmb().to(DEVICE)

train_loader = DataLoader(AffectDataset(train_df), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(AffectDataset(val_df), batch_size=BATCH_SIZE)

optimizer = AdamW(model.parameters(), lr=LR)
criterion = nn.SmoothL1Loss(beta=0.5)  # HUBER

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parametri allenabili: {trainable_params}")

best_score = -1

for epoch in range(1, EPOCHS + 1):

    model.train()
    total_loss = 0.0

    for b in train_loader:
        optimizer.zero_grad()
        preds = model(
            b["input_ids"].to(DEVICE),
            b["attention_mask"].to(DEVICE),
            b["user_idx"].to(DEVICE)
        )
        loss = criterion(preds, b["labels"].to(DEVICE))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # ---- VALIDATION ----
    model.eval()
    P, Y, U = [], [], []

    with torch.no_grad():
        for b in val_loader:
            p = model(
                b["input_ids"].to(DEVICE),
                b["attention_mask"].to(DEVICE),
                b["user_idx"].to(DEVICE)
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
        patience = 0
        torch.save(model.state_dict(), SAVE_PATH)
        print(" → Saved best model")


# ------------------------
# FINAL INFERENCE
# ------------------------
model.load_state_dict(torch.load(SAVE_PATH))
model.eval()

test_df = pd.read_csv(TEST_PATH)
test_df["text"] = test_df["text"].fillna("").astype(str)

test_loader = DataLoader(
    AffectTestDataset(test_df, user_to_idx),
    batch_size=BATCH_SIZE
)

rows = []

with torch.no_grad():
    for b in test_loader:
        preds = model(
            b["input_ids"].to(DEVICE),
            b["attention_mask"].to(DEVICE),
            b["user_idx"].to(DEVICE)
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
print("Best avg r_composite:", best_score)
