#!/usr/bin/env python3
"""
Unified runner for Subtask 2A / 2B.

Usage:
  python run_subtask2.py -a [--train_path ... --test_path ... --out_path ...]
  python run_subtask2.py -b [--train_path ... --test_path ... --out_path ...]
"""

import argparse
import os
import csv
import math
import random
from dataclasses import dataclass
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error() 
from torch.optim import AdamW

# OPTIONAL: Import eval function if available
try:
    from eval import task2_correlation
except Exception:
    task2_correlation = None
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets"


# ==========================================================
# Common utilities
# ==========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def _safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


# ==========================================================
# -------------------------- TASK 2A ------------------------
# ==========================================================
@dataclass
class Task2AConfig:
    train_path: str
    test_path: str
    out_path: str
    log_file: str

    model_name: str = "roberta-base"
    max_len: int = 320
    batch_size: int = 4
    lr: float = 1e-5
    epochs: int = 16
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 100

    k_history: int = 4
    huber_beta: float = 0.3


def add_time_features_2a(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour.fillna(-1).astype(int)
    df["dow"] = ts.dt.dayofweek.fillna(-1).astype(int)
    df["month"] = ts.dt.month.fillna(-1).astype(int)
    return df


def format_event_row_2a(row: pd.Series) -> str:
    hour = _safe_int(row.get("hour", -1), -1)
    dow = _safe_int(row.get("dow", -1), -1)
    month = _safe_int(row.get("month", -1), -1)
    phase = row.get("collection_phase", "UNK")
    is_words = _safe_int(row.get("is_words", 0), 0)

    meta = f"[H={hour} DOW={dow} M={month} PHASE={phase} WORDS={is_words}]"

    v = _safe_float(row.get("valence", np.nan), np.nan)
    a = _safe_float(row.get("arousal", np.nan), np.nan)
    if np.isnan(v) or np.isnan(a):
        va = "V=UNK A=UNK"
    else:
        va = f"V={v:.3f} A={a:.3f}"

    txt = (row.get("text", "") or "").replace("\n", " ").strip()
    if txt == "":
        txt = "[NO_TEXT]"

    typ = "[FEELING_WORDS]" if is_words == 1 else "[ESSAY]"
    return f"{meta} {typ} {va} | {txt}"


def build_context_text_2a(df: pd.DataFrame, k: int = 4) -> pd.DataFrame:
    df = add_time_features_2a(df)
    df = df.copy()

    sort_cols = ["user_id", "timestamp"]
    if "text_id" in df.columns:
        sort_cols.append("text_id")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    contexts = [""] * len(df)
    for _, idxs in df.groupby("user_id").groups.items():
        idxs = list(idxs)
        window = deque(maxlen=k)
        for i in idxs:
            row = df.loc[i]
            window.append(format_event_row_2a(row))
            contexts[i] = "\n---\n".join(window)

    df["context_text"] = contexts
    return df


def temporal_split_per_user_2a(df: pd.DataFrame, val_frac: float = 0.2, min_user_events: int = 6):
    train_parts = []
    val_parts = []
    sort_cols = ["timestamp"]
    if "text_id" in df.columns:
        sort_cols.append("text_id")

    for _, g in df.groupby("user_id"):
        g = g.sort_values(sort_cols)
        n = len(g)
        if n < min_user_events:
            train_parts.append(g)
            continue
        cut = int(np.floor((1 - val_frac) * n))
        train_parts.append(g.iloc[:cut])
        val_parts.append(g.iloc[cut:])

    train_split = pd.concat(train_parts).reset_index(drop=True)
    val_split = pd.concat(val_parts).reset_index(drop=True) if val_parts else df.iloc[:0].copy()
    return train_split, val_split


def run_task2a(cfg: Task2AConfig) -> None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TARGET_COLS = ["state_change_valence", "state_change_arousal"]

    set_seed(cfg.seed)
    print("[2A] Device:", DEVICE)
    print("[2A] Train:", cfg.train_path)
    print("[2A] Test :", cfg.test_path)

    # Load training
    train_df = pd.read_csv(cfg.train_path)
    train_df["text"] = train_df["text"].fillna("")
    train_df = train_df.dropna(subset=TARGET_COLS).reset_index(drop=True)

    train_df = build_context_text_2a(train_df, k=cfg.k_history)
    train_split, val_split = temporal_split_per_user_2a(train_df, val_frac=0.2)
    print("[2A] Train/Val sizes:", len(train_split), len(val_split))

    # Normalize targets
    y_tr_orig = train_split[TARGET_COLS].astype(np.float32).values
    mu = y_tr_orig.mean(axis=0)
    sigma = y_tr_orig.std(axis=0) + 1e-8

    delta_v_min, delta_v_max = float(y_tr_orig[:, 0].min()), float(y_tr_orig[:, 0].max())
    delta_a_min, delta_a_max = float(y_tr_orig[:, 1].min()), float(y_tr_orig[:, 1].max())
    print("[2A] TRAIN target stats mu/sigma:", mu, sigma)

    train_split = train_split.copy()
    val_split = val_split.copy()
    train_split[TARGET_COLS] = (train_split[TARGET_COLS].astype(np.float32).values - mu) / sigma
    if len(val_split) > 0:
        val_split[TARGET_COLS] = (val_split[TARGET_COLS].astype(np.float32).values - mu) / sigma

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    class AffectDataset(Dataset):
        def __init__(self, df: pd.DataFrame):
            self.df = df.reset_index(drop=True)
            self.texts = self.df["context_text"].fillna("").tolist()
            self.user_ids = self.df["user_id"].tolist()
            self.y = self.df[TARGET_COLS].values.astype(np.float32)

        def __len__(self): return len(self.df)

        def __getitem__(self, idx: int):
            enc = tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=cfg.max_len,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.y[idx])
            item["user_id"] = self.user_ids[idx]
            return item

    class AffectTestDataset(Dataset):
        def __init__(self, df: pd.DataFrame):
            self.df = df.reset_index(drop=True)
            self.texts = self.df["context_text"].fillna("").tolist()
            self.user_ids = self.df["user_id"].tolist()

        def __len__(self): return len(self.df)

        def __getitem__(self, idx: int):
            enc = tokenizer(
                self.texts[idx],
                truncation=True,
                padding="max_length",
                max_length=cfg.max_len,
                return_tensors="pt",
            )
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["user_id"] = self.user_ids[idx]
            return item

    def collate_fn(batch):
        out = {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
            "user_id": [b["user_id"] for b in batch],
        }
        if "token_type_ids" in batch[0]:
            out["token_type_ids"] = torch.stack([b["token_type_ids"] for b in batch])
        return out

    train_loader = DataLoader(AffectDataset(train_split), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(AffectDataset(val_split), batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn) if len(val_split) > 0 else None

    # Model
    config = AutoConfig.from_pretrained(cfg.model_name)
    config.num_labels = 2
    config.problem_type = "regression"
    config.hidden_dropout_prob = 0.2
    config.attention_probs_dropout_prob = 0.2

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        config=config,
        ignore_mismatched_sizes=True
    ).to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    num_training_steps = cfg.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    huber = torch.nn.SmoothL1Loss(reduction="none", beta=cfg.huber_beta)
    dim_weights = torch.tensor([1.0, 1.0], device=DEVICE)

    def custom_loss(preds: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        per_dim = huber(preds, labels) * dim_weights
        return per_dim.mean()

    @torch.no_grad()
    def eval_loss(loader):
        model.eval()
        total, n = 0.0, 0
        for batch in loader:
            batch.pop("user_id")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**{k: v for k, v in batch.items() if k != "labels"})
            loss = custom_loss(out.logits, batch["labels"])
            total += float(loss.item())
            n += 1
        return total / max(1, n)

    best_val = 1e18
    best_state = None

    # Train
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            batch.pop("user_id")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            out = model(**{k: v for k, v in batch.items() if k != "labels"})
            loss = custom_loss(out.logits, batch["labels"])
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += float(loss.item())

        train_loss = total_loss / max(1, len(train_loader))

        if val_loader is not None:
            vloss = eval_loss(val_loader)
            print(f"[2A] Epoch {epoch} | train_loss={train_loss:.4f} | val_loss={vloss:.4f}")
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            print(f"[2A] Epoch {epoch} | train_loss={train_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(DEVICE)
        print("[2A] Loaded best model by val_loss:", best_val)

    # Predict on test
    test_df = pd.read_csv(cfg.test_path)
    test_df["text"] = test_df["text"].fillna("")
    test_df = build_context_text_2a(test_df, k=cfg.k_history)
    test_loader = DataLoader(AffectTestDataset(test_df), batch_size=cfg.batch_size, shuffle=False)

    @torch.no_grad()
    def predict(loader) -> Tuple[np.ndarray, np.ndarray]:
        model.eval()
        all_preds = []
        all_user_ids: List[str] = []

        for batch in loader:
            user_ids = batch.pop("user_id")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out = model(**batch)

            preds_norm = out.logits.detach().cpu().numpy().astype(np.float32)
            all_preds.append(preds_norm)
            all_user_ids.extend(user_ids)

        preds_norm = np.vstack(all_preds)
        preds = preds_norm * sigma + mu

        preds[:, 0] = np.clip(preds[:, 0], delta_v_min, delta_v_max)
        preds[:, 1] = np.clip(preds[:, 1], delta_a_min, delta_a_max)
        return np.array(all_user_ids), preds

    user_ids, preds = predict(test_loader)

    submission = pd.DataFrame({
        "user_id": user_ids,
        "state_change_valence": preds[:, 0],
        "state_change_arousal": preds[:, 1],
    })
    submission.to_csv(cfg.out_path, index=False)
    print("[2A] Saved submission:", cfg.out_path)

    # Optional scoring + logging
    if task2_correlation is not None and set(TARGET_COLS).issubset(set(test_df.columns)):
        scored = test_df.dropna(subset=TARGET_COLS).reset_index(drop=True)
        if len(scored) >= 2:
            scored_loader = DataLoader(AffectTestDataset(scored), batch_size=cfg.batch_size, shuffle=False)
            scored_user_ids, scored_preds = predict(scored_loader)

            gold_v = scored[TARGET_COLS[0]].astype(np.float32).values
            gold_a = scored[TARGET_COLS[1]].astype(np.float32).values

            corr_v = task2_correlation(scored_user_ids, scored_preds[:, 0], gold_v)
            corr_a = task2_correlation(scored_user_ids, scored_preds[:, 1], gold_a)

            print("\n[2A UNOFFICIAL] task2_correlation on TEST rows with gold:")
            print("VALENCE:", corr_v)
            print("AROUSAL:", corr_a)

            log_data = {
                "Model": cfg.model_name,
                "Seed": cfg.seed,
                "History_K": cfg.k_history,
                "Epochs": cfg.epochs,
                "Batch_Size": cfg.batch_size,
                "Learning_Rate": cfg.lr,
                "Best_Val_Loss": round(best_val, 4) if isinstance(best_val, float) else "N/A",
                "Val_R": corr_v.get("r"),
                "Val_P": corr_v.get("p"),
                "Val_MAE": corr_v.get("mae"),
                "Aro_R": corr_a.get("r"),
                "Aro_P": corr_a.get("p"),
                "Aro_MAE": corr_a.get("mae"),
            }

            file_exists = os.path.isfile(cfg.log_file)
            with open(cfg.log_file, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=log_data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(log_data)

            print(f"[2A] Logged results to: {cfg.log_file}")
        else:
            print("[2A] No usable gold rows in TEST for correlation.")
    else:
        print("[2A] No gold columns in TEST or task2_correlation unavailable.")


# ==========================================================
# -------------------------- TASK 2B ------------------------
# ==========================================================
@dataclass
class Task2BConfig:
    train_path: str
    test_path: str
    out_path: str

    model_name: str = "mental/mental-roberta-base"
    seed: int = 200

    # text sampling
    k_texts: int = 6
    text_sampling: str = "LAST_K"
    max_len: int = 256

    # optimization
    batch_size: int = 8
    grad_accum_steps: int = 2
    epochs: int = 50
    patience: int = 8
    grad_clip: float = 1.0

    lr_head: float = 5e-4
    lr_backbone: float = 1e-5
    unfreeze_last_n_layers: int = 3

    wd_head: float = 0.01
    wd_backbone: float = 0.0

    dropout: float = 0.2

    # CV
    n_splits: int = 4


def safe_float_2b(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def sort_user_2b(g: pd.DataFrame) -> pd.DataFrame:
    cols = ["timestamp"]
    if "text_id" in g.columns:
        cols.append("text_id")
    return g.sort_values(cols).reset_index(drop=True)


def sample_group1_texts_2b(g1: pd.DataFrame, k: int, mode: str):
    texts = g1["text"].fillna("").astype(str).tolist()
    n = len(texts)
    if n == 0:
        return ["[NO_TEXT]"]
    if n <= k:
        return [t if t.strip() else "[NO_TEXT]" for t in texts]

    if mode == "LAST_K":
        chosen = texts[-k:]
    elif mode == "UNIFORM_K":
        idx = np.linspace(0, n - 1, num=k, dtype=int)
        chosen = [texts[i] for i in idx]
    else:
        raise ValueError(f"Unknown TEXT_SAMPLING={mode}")

    return [t if t.strip() else "[NO_TEXT]" for t in chosen]


def _slope_over_time_2b(t_seconds: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(t_seconds) & np.isfinite(y)
    t = t_seconds[mask]
    yy = y[mask]
    if len(yy) < 2:
        return 0.0
    t = t - t.mean()
    denom = float(np.sum(t * t))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(t * (yy - yy.mean())) / denom)


def compute_numeric_features_group1_2b(g1: pd.DataFrame) -> np.ndarray:
    v = g1["valence"].apply(safe_float_2b).to_numpy(dtype=np.float32)
    a = g1["arousal"].apply(safe_float_2b).to_numpy(dtype=np.float32)

    def stats_mean_std(arr):
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return 0.0, 0.0
        return float(arr.mean()), float(arr.std())

    v_mean, v_std = stats_mean_std(v)
    a_mean, a_std = stats_mean_std(a)

    n_texts = int(len(g1))

    span_days = 0.0
    if n_texts >= 2:
        tmin, tmax = g1["timestamp"].min(), g1["timestamp"].max()
        if pd.notna(tmin) and pd.notna(tmax):
            span_days = float((tmax - tmin).total_seconds() / 86400.0)

    v_delta = 0.0
    a_delta = 0.0
    if n_texts >= 2:
        v0 = safe_float_2b(g1["valence"].iloc[0], default=np.nan)
        v1 = safe_float_2b(g1["valence"].iloc[-1], default=np.nan)
        a0 = safe_float_2b(g1["arousal"].iloc[0], default=np.nan)
        a1 = safe_float_2b(g1["arousal"].iloc[-1], default=np.nan)
        if np.isfinite(v0) and np.isfinite(v1):
            v_delta = float(v1 - v0)
        if np.isfinite(a0) and np.isfinite(a1):
            a_delta = float(a1 - a0)

    t = pd.to_datetime(g1["timestamp"], errors="coerce")
    t0 = t.min()
    if pd.isna(t0):
        t_seconds = np.zeros(len(g1), dtype=np.float32)
    else:
        t_seconds = ((t - t0).dt.total_seconds()).to_numpy(dtype=np.float32)

    v_slope = _slope_over_time_2b(t_seconds, v.astype(np.float32))
    a_slope = _slope_over_time_2b(t_seconds, a.astype(np.float32))

    return np.array(
        [
            v_mean, v_std,
            a_mean, a_std,
            float(n_texts),
            float(span_days),
            float(v_delta),
            float(a_delta),
            float(v_slope),
            float(a_slope),
        ],
        dtype=np.float32,
    )


BASE_FEAT_DIM_2B = 10
NUM_FEAT_DIM_2B = BASE_FEAT_DIM_2B
TARGET_COLS_2B = ["disposition_change_valence", "disposition_change_arousal"]


def build_userlevel_2b(df: pd.DataFrame, is_train: bool, k_texts: int, sampling: str) -> pd.DataFrame:
    d = df.copy()

    needed = {"user_id", "text", "timestamp", "group"}
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise ValueError(f"Missing required columns in input df: {missing}")

    d["text"] = d["text"].fillna("").astype(str)
    d["timestamp"] = pd.to_datetime(d["timestamp"], errors="coerce")
    d["group"] = pd.to_numeric(d["group"], errors="coerce")

    rows = []
    for uid, g in d.groupby("user_id"):
        g = sort_user_2b(g)
        g1 = g[g["group"] == 1]
        if len(g1) == 0:
            continue

        sampled = sample_group1_texts_2b(g1, k_texts, sampling)
        text_block = " </s> ".join([t.strip().replace("\n", " ") for t in sampled])

        num_feats = compute_numeric_features_group1_2b(g1)

        row = {
            "user_id": uid,
            "input_text": text_block,
            "num_feats": num_feats,
        }

        if is_train:
            if TARGET_COLS_2B[0] not in g.columns or TARGET_COLS_2B[1] not in g.columns:
                raise ValueError(
                    f"Train df must include targets: {TARGET_COLS_2B}. "
                    f"Missing: {[c for c in TARGET_COLS_2B if c not in g.columns]}"
                )
            row[TARGET_COLS_2B[0]] = safe_float_2b(g[TARGET_COLS_2B[0]].iloc[0])
            row[TARGET_COLS_2B[1]] = safe_float_2b(g[TARGET_COLS_2B[1]].iloc[0])

        rows.append(row)

    return pd.DataFrame(rows)


class FusionDataset2B(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int, with_labels: bool):
        self.df = df.reset_index(drop=True)
        self.texts = self.df["input_text"].fillna("").tolist()
        self.user_ids = self.df["user_id"].tolist()
        self.num_feats = np.stack(self.df["num_feats"].to_numpy()).astype(np.float32)
        self.with_labels = with_labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        if with_labels:
            self.y = self.df[TARGET_COLS_2B].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["user_id"] = self.user_ids[idx]
        item["num_feats"] = torch.tensor(self.num_feats[idx], dtype=torch.float32)
        if self.with_labels:
            item["labels"] = torch.tensor(self.y[idx], dtype=torch.float32)
        return item


def collate_fn_2b(batch):
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "num_feats": torch.stack([b["num_feats"] for b in batch]),
        "user_id": [b["user_id"] for b in batch],
    }
    if "labels" in batch[0]:
        out["labels"] = torch.stack([b["labels"] for b in batch])
    return out


class FusionRegressor2B(nn.Module):
    def __init__(self, model_name: str, num_feat_dim: int, dropout: float):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size

        self.num_proj = nn.Sequential(
            nn.Linear(num_feat_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(hidden + 32, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

        self._init_weights(self.num_proj)
        self._init_weights(self.head)

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input_ids, attention_mask, num_feats):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).type_as(last)
        sum_emb = (last * mask).sum(dim=1)
        len_emb = mask.sum(dim=1).clamp(min=1e-6)
        pool = sum_emb / len_emb
        cls_emb = pool
        nf = self.num_proj(num_feats)
        x = torch.cat([cls_emb, nf], dim=1)
        return self.head(x)


def set_optimization_params_2b(model, lr_backbone, lr_head, wd_head, wd_backbone, unfreeze_n):
    for p in model.backbone.parameters():
        p.requires_grad = False

    if unfreeze_n and unfreeze_n > 0:
        layers = model.backbone.encoder.layer
        for layer in layers[-unfreeze_n:]:
            for p in layer.parameters():
                p.requires_grad = True

        for name, p in model.backbone.named_parameters():
            if "LayerNorm" in name:
                p.requires_grad = True

    groups = []
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if lr_backbone > 0 and backbone_params:
        groups.append({"params": backbone_params, "lr": lr_backbone, "weight_decay": wd_backbone})

    groups.append({"params": model.num_proj.parameters(), "lr": lr_head, "weight_decay": wd_head})
    groups.append({"params": model.head.parameters(), "lr": lr_head, "weight_decay": wd_head})
    return groups


def train_one_fold_2b(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer,
    cfg: Task2BConfig,
    device: str,
):
    # fold-only target normalization
    y_tr = train_df[TARGET_COLS_2B].values.astype(np.float32)
    mu = y_tr.mean(axis=0).astype(np.float32)
    sigma = (y_tr.std(axis=0) + 1e-6).astype(np.float32)

    tr = train_df.copy()
    va = val_df.copy()
    tr[TARGET_COLS_2B] = (tr[TARGET_COLS_2B].values.astype(np.float32) - mu) / sigma
    va[TARGET_COLS_2B] = (va[TARGET_COLS_2B].values.astype(np.float32) - mu) / sigma

    # fold-only numeric normalization
    X_tr = np.stack(tr["num_feats"].to_numpy()).astype(np.float32)
    xf_mu = X_tr.mean(axis=0).astype(np.float32)
    xf_sd = (X_tr.std(axis=0) + 1e-6).astype(np.float32)

    def norm_feats_df(df_in):
        arr = np.stack(df_in["num_feats"].to_numpy()).astype(np.float32)
        arr = (arr - xf_mu) / xf_sd
        out = df_in.copy()
        out["num_feats"] = list(arr)
        return out

    tr = norm_feats_df(tr)
    va = norm_feats_df(va)

    train_loader = DataLoader(
        FusionDataset2B(tr, tokenizer, max_len=cfg.max_len, with_labels=True),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn_2b,
    )
    val_loader = DataLoader(
        FusionDataset2B(va, tokenizer, max_len=cfg.max_len, with_labels=True),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn_2b,
    )

    model = FusionRegressor2B(cfg.model_name, NUM_FEAT_DIM_2B, cfg.dropout).to(device)
    optim_params = set_optimization_params_2b(
        model,
        cfg.lr_backbone,
        cfg.lr_head,
        cfg.wd_head,
        cfg.wd_backbone,
        cfg.unfreeze_last_n_layers,
    )
    optimizer = AdamW(optim_params)

    steps_per_epoch = math.ceil(len(train_loader) / cfg.grad_accum_steps)
    num_training_steps = max(1, steps_per_epoch * cfg.epochs)
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    loss_fn = nn.SmoothL1Loss(reduction="none", beta=1.0)
    W = torch.tensor([1.5, 1.0], device=device)

    best_val_mae = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        train_loss_sum = 0.0
        train_items = 0

        for step, batch in enumerate(train_loader):
            preds = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["num_feats"].to(device),
            )
            labels = batch["labels"].to(device)

            loss_per_elem = loss_fn(preds, labels)  # [B,2]
            loss_per_sample = (loss_per_elem * W).mean(dim=1)  # [B]
            loss = loss_per_sample.mean() / cfg.grad_accum_steps
            loss.backward()

            if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            bs = labels.size(0)
            train_loss_sum += float(loss_per_sample.detach().mean().item()) * bs
            train_items += bs

        train_loss_epoch = train_loss_sum / max(1, train_items)

        # validation
        model.eval()
        val_loss_sum = 0.0
        val_items = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                preds = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["num_feats"].to(device),
                )
                labels = batch["labels"].to(device)

                loss_per_elem = loss_fn(preds, labels)
                loss_per_sample = (loss_per_elem * W).mean(dim=1)
                bs = labels.size(0)
                val_loss_sum += float(loss_per_sample.mean().item()) * bs
                val_items += bs

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        val_loss_epoch = val_loss_sum / max(1, val_items)

        p_norm = np.vstack(all_preds).astype(np.float32)
        y_norm = np.vstack(all_labels).astype(np.float32)

        p_orig = p_norm * sigma + mu
        y_orig = y_norm * sigma + mu

        mae_v = float(np.mean(np.abs(p_orig[:, 0] - y_orig[:, 0])))
        mae_a = float(np.mean(np.abs(p_orig[:, 1] - y_orig[:, 1])))
        mae_avg = (mae_v + mae_a) / 2.0

        improved = mae_avg + 1e-12 < best_val_mae
        if improved:
            best_val_mae = mae_avg
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"  Ep {epoch:02d}: "
            f"Tr_L={train_loss_epoch:.4f} | "
            f"Val_L={val_loss_epoch:.4f} | "
            f"Val_MAE={mae_avg:.4f} (V={mae_v:.4f}, A={mae_a:.4f}) | "
            f"best={best_val_mae:.4f}"
            + (" ✅" if improved else "")
        )

        if patience_counter >= cfg.patience:
            print(f"  🛑 Early Stopping! No improvement for {cfg.patience} epochs.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    scalers = {"mu": mu, "sigma": sigma, "xf_mu": xf_mu, "xf_sd": xf_sd}
    return model, scalers, best_val_mae


@torch.no_grad()
def predict_with_scalers_2b(model, df_userlevel, tokenizer, sc, cfg: Task2BConfig, device: str):
    arr = np.stack(df_userlevel["num_feats"].to_numpy()).astype(np.float32)
    arr = (arr - sc["xf_mu"]) / sc["xf_sd"]
    df_norm = df_userlevel.copy()
    df_norm["num_feats"] = list(arr)

    loader = DataLoader(
        FusionDataset2B(df_norm, tokenizer, max_len=cfg.max_len, with_labels=False),
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn_2b,
    )

    model.eval()
    preds = []
    for batch in loader:
        p = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["num_feats"].to(device),
        )
        preds.append(p.cpu().numpy().astype(np.float32))

    p_norm = np.vstack(preds)
    p_orig = p_norm * sc["sigma"] + sc["mu"]
    return p_orig.astype(np.float32)


def run_task2b(cfg: Task2BConfig) -> None:
    # imports here so 2A can run without sklearn installed (if needed)
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import Ridge

    # quiet HF
    import warnings
    from transformers.utils import logging as hf_logging
    warnings.filterwarnings("ignore", category=UserWarning)
    hf_logging.set_verbosity_error()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg.seed)

    print(f"[2B] DEVICE: {DEVICE} | MODEL: {cfg.model_name}")
    print(f"[2B] TRAIN_PATH: {cfg.train_path}")
    print(f"[2B] TEST_PATH : {cfg.test_path}")
    print(f"[2B] NUM_FEAT_DIM: {NUM_FEAT_DIM_2B}")

    # ----- Load train -----
    train_raw = pd.read_csv(cfg.train_path)
    train_users = build_userlevel_2b(train_raw, is_train=True, k_texts=cfg.k_texts, sampling=cfg.text_sampling).reset_index(drop=True)
    train_users = train_users.dropna(subset=TARGET_COLS_2B).reset_index(drop=True)
    print(f"[2B] Train users: {len(train_users)} | unique: {train_users['user_id'].nunique()}")

    # ----- Load test -----
    test_raw = pd.read_csv(cfg.test_path)

    if "is_forecasting_user" in test_raw.columns:
        n_true = int((test_raw["is_forecasting_user"] == True).sum())
        print(f"[2B] is_forecasting_user==True rows: {n_true}")
        test_raw = test_raw[test_raw["is_forecasting_user"] == True].reset_index(drop=True)
        if len(test_raw) == 0:
            raise ValueError("[2B] No forecasting users after filter — check TEST_PATH / file content.")

    test_users = build_userlevel_2b(test_raw, is_train=False, k_texts=cfg.k_texts, sampling=cfg.text_sampling).reset_index(drop=True)
    print(f"[2B] Test users: {len(test_users)} | unique: {test_users['user_id'].nunique()}")

    # ----- Tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    # ----- CV -----
    gkf = GroupKFold(n_splits=cfg.n_splits)

    oof_preds = np.zeros((len(train_users), 2), dtype=np.float32)
    test_preds_folds = []
    fold_scores = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(train_users, groups=train_users["user_id"]), 1):
        print(f"\n===== [2B] FOLD {fold}/{cfg.n_splits} =====")
        tr_df = train_users.iloc[tr_idx].reset_index(drop=True)
        va_df = train_users.iloc[va_idx].reset_index(drop=True)

        model, sc, best = train_one_fold_2b(tr_df, va_df, tokenizer, cfg, DEVICE)
        fold_scores.append(best)

        va_pred = predict_with_scalers_2b(model, va_df, tokenizer, sc, cfg, DEVICE)
        oof_preds[va_idx] = va_pred

        te_pred = predict_with_scalers_2b(model, test_users, tokenizer, sc, cfg, DEVICE)
        test_preds_folds.append(te_pred)

        del model
        torch.cuda.empty_cache()

    # ----- Global calibration (Ridge) on OOF -----
    print("\n===== [2B] GLOBAL CALIBRATION (Ridge) =====")
    y_true = train_users[TARGET_COLS_2B].values.astype(np.float32)

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_v = None
    best_a = None
    best_mae_v = 1e9
    best_mae_a = 1e9

    for a in alphas:
        m = Ridge(alpha=a).fit(oof_preds[:, [0]], y_true[:, 0])
        pv = m.predict(oof_preds[:, [0]])
        mae = np.mean(np.abs(pv - y_true[:, 0]))
        if mae < best_mae_v:
            best_mae_v = mae
            best_v = a

    for a in alphas:
        m = Ridge(alpha=a).fit(oof_preds[:, [1]], y_true[:, 1])
        pa = m.predict(oof_preds[:, [1]])
        mae = np.mean(np.abs(pa - y_true[:, 1]))
        if mae < best_mae_a:
            best_mae_a = mae
            best_a = a

    print("[2B] Best alpha V:", best_v, "| Best alpha A:", best_a)

    cal_v = Ridge(alpha=best_v).fit(oof_preds[:, [0]], y_true[:, 0])
    cal_a = Ridge(alpha=best_a).fit(oof_preds[:, [1]], y_true[:, 1])

    oof_cal = oof_preds.copy()
    oof_cal[:, 0] = cal_v.predict(oof_preds[:, [0]])
    oof_cal[:, 1] = cal_a.predict(oof_preds[:, [1]])

    mae_v = float(np.mean(np.abs(oof_cal[:, 0] - y_true[:, 0])))
    mae_a = float(np.mean(np.abs(oof_cal[:, 1] - y_true[:, 1])))
    mae_avg = (mae_v + mae_a) / 2.0
    print(f"[2B] OOF MAE (cal): avg={mae_avg:.4f} | V={mae_v:.4f} | A={mae_a:.4f}")
    print(f"[2B] Fold best MAE (raw): {[round(x,4) for x in fold_scores]}")

    # ----- Ensemble test (mean) + calibration -----
    avg_test = np.mean(np.stack(test_preds_folds, axis=0), axis=0).astype(np.float32)

    final_test = avg_test.copy()
    final_test[:, 0] = cal_v.predict(avg_test[:, [0]])
    final_test[:, 1] = cal_a.predict(avg_test[:, [1]])

    submission = pd.DataFrame({
        "user_id": test_users["user_id"].values,
        TARGET_COLS_2B[0]: final_test[:, 0],
        TARGET_COLS_2B[1]: final_test[:, 1],
    })
    submission.to_csv(cfg.out_path, index=False)
    print(f"\n[2B] Saved: {cfg.out_path}")
    print(submission.head())

    # ----- Optional unofficial scoring if gold present in test file -----
    if task2_correlation is not None and set(TARGET_COLS_2B).issubset(set(test_raw.columns)):
        print("\n===== [2B] UNOFFICIAL SCORING (TEST) =====")
        gold = test_raw.dropna(subset=TARGET_COLS_2B).groupby("user_id", as_index=False)[TARGET_COLS_2B].mean()
        merged = gold.merge(submission, on="user_id", suffixes=("_gold", "_pred"))

        if len(merged) > 1:
            v_gold = merged[f"{TARGET_COLS_2B[0]}_gold"]
            v_pred = merged[f"{TARGET_COLS_2B[0]}_pred"]
            a_gold = merged[f"{TARGET_COLS_2B[1]}_gold"]
            a_pred = merged[f"{TARGET_COLS_2B[1]}_pred"]

            corr_v = task2_correlation(merged["user_id"], v_pred, v_gold)
            corr_a = task2_correlation(merged["user_id"], a_pred, a_gold)

            print("VALENCE:", corr_v)
            print("AROUSAL:", corr_a)
        else:
            print("[2B] Not enough samples for correlation.")
    else:
        print("\n[2B] Scoring skipped: gold columns missing or task2_correlation not imported.")


# ==========================================================
# CLI
# ==========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Subtask 2A or 2B with -a / -b.")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-a", "--task2a", action="store_true", help="Run Subtask 2A")
    group.add_argument("-b", "--task2b", action="store_true", help="Run Subtask 2B")

    # shared-ish
    parser.add_argument("--seed", type=int, default=None, help="Override default seed (2A=100, 2B=200)")

    # -------------------------
    # 2A paths (robusti)
    # -------------------------
    parser.add_argument(
        "--train_path_2a",
        type=str,
        default=str(DATA_DIR / "train_subtask2a.csv"),
    )
    parser.add_argument(
        "--test_path_2a",
        type=str,
        default=str(DATA_DIR / "subtask2a_forecasting_user_marker.csv"),
    )
    parser.add_argument("--out_path_2a", type=str, default="subtask2a_predictions.csv")
    parser.add_argument("--log_file_2a", type=str, default="experiment_results_log.csv")

    # 2A hparams
    parser.add_argument("--model_name_2a", type=str, default="roberta-base")
    parser.add_argument("--max_len_2a", type=int, default=320)
    parser.add_argument("--batch_size_2a", type=int, default=4)
    parser.add_argument("--lr_2a", type=float, default=1e-5)
    parser.add_argument("--epochs_2a", type=int, default=16)
    parser.add_argument("--weight_decay_2a", type=float, default=0.01)
    parser.add_argument("--grad_clip_2a", type=float, default=1.0)
    parser.add_argument("--k_history_2a", type=int, default=4)
    parser.add_argument("--huber_beta_2a", type=float, default=0.3)

    # -------------------------
    # 2B paths (robusti)
    # -------------------------
    parser.add_argument(
        "--train_path_2b",
        type=str,
        default=str(DATA_DIR / "train_subtask2b_detailed.csv"),
    )
    parser.add_argument(
        "--test_path_2b",
        type=str,
        default=str(DATA_DIR / "subtask2b_forecasting_user_marker.csv"),
    )
    parser.add_argument("--out_path_2b", type=str, default="subtask2b_results.csv")

    # 2B hparams
    parser.add_argument("--model_name_2b", type=str, default="mental/mental-roberta-base")
    parser.add_argument("--k_texts_2b", type=int, default=6)
    parser.add_argument("--text_sampling_2b", type=str, default="LAST_K", choices=["LAST_K", "UNIFORM_K"])
    parser.add_argument("--max_len_2b", type=int, default=256)
    parser.add_argument("--batch_size_2b", type=int, default=8)
    parser.add_argument("--grad_accum_steps_2b", type=int, default=2)
    parser.add_argument("--epochs_2b", type=int, default=50)
    parser.add_argument("--patience_2b", type=int, default=8)
    parser.add_argument("--grad_clip_2b", type=float, default=1.0)
    parser.add_argument("--lr_head_2b", type=float, default=5e-4)
    parser.add_argument("--lr_backbone_2b", type=float, default=1e-5)
    parser.add_argument("--unfreeze_last_n_layers_2b", type=int, default=3)
    parser.add_argument("--wd_head_2b", type=float, default=0.01)
    parser.add_argument("--wd_backbone_2b", type=float, default=0.0)
    parser.add_argument("--dropout_2b", type=float, default=0.2)
    parser.add_argument("--n_splits_2b", type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.task2a:
        seed = args.seed if args.seed is not None else 100
        cfg = Task2AConfig(
            train_path=args.train_path_2a,
            test_path=args.test_path_2a,
            out_path=args.out_path_2a,
            log_file=args.log_file_2a,
            model_name=args.model_name_2a,
            max_len=args.max_len_2a,
            batch_size=args.batch_size_2a,
            lr=args.lr_2a,
            epochs=args.epochs_2a,
            weight_decay=args.weight_decay_2a,
            grad_clip=args.grad_clip_2a,
            seed=seed,
            k_history=args.k_history_2a,
            huber_beta=args.huber_beta_2a,
        )
        run_task2a(cfg)
        return

    if args.task2b:
        seed = args.seed if args.seed is not None else 200
        cfg = Task2BConfig(
            train_path=args.train_path_2b,
            test_path=args.test_path_2b,
            out_path=args.out_path_2b,
            model_name=args.model_name_2b,
            seed=seed,
            k_texts=args.k_texts_2b,
            text_sampling=args.text_sampling_2b,
            max_len=args.max_len_2b,
            batch_size=args.batch_size_2b,
            grad_accum_steps=args.grad_accum_steps_2b,
            epochs=args.epochs_2b,
            patience=args.patience_2b,
            grad_clip=args.grad_clip_2b,
            lr_head=args.lr_head_2b,
            lr_backbone=args.lr_backbone_2b,
            unfreeze_last_n_layers=args.unfreeze_last_n_layers_2b,
            wd_head=args.wd_head_2b,
            wd_backbone=args.wd_backbone_2b,
            dropout=args.dropout_2b,
            n_splits=args.n_splits_2b,
        )
        run_task2b(cfg)
        return


if __name__ == "__main__":
    main()
