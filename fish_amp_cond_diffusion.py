"""
Fish AMP Conditional Discrete Diffusion (single file, PyTorch)
=============================================================
This script implements a **discrete diffusion** baseline for peptide sequence generation with
**gene-expression conditioning**. It supports a two-stage workflow: pretrain on general AMP,
then finetune on fish-specific data with expression vectors as conditions.

Subcommands
-----------
1) prep   : clean CSV, (optional) merge expression table, standardize expression columns
2) train  : train diffusion model (discrete, absorbing MASK corruption, predict x0)
3) sample : generate peptides with expression-guided CFG sampling

Minimal CSV schema (after prep)
-------------------------------
- sequence : str, peptide in {A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y,X}
- (optional) active : 0/1
- (optional) hemolytic : 0/1
- (optional) gene_id : for merging expression table
- expr_* : standardized expression features (continuous), or specify with --expr-cols

Examples
--------
# 1) Preprocess (merge expression by gene_id and standardize)
python fish_amp_cond_diffusion.py prep \
  --csv data/fish_amp.csv \
  --out data/fish_amp_clean.csv \
  --min-len 8 --max-len 60 \
  --expr-table data/expr_matrix.csv --csv-key gene_id --expr-key gene_id \
  --expr-cols expr_brain,expr_eye,expr_gill,expr_muscle,expr_heart,expr_intestine,expr_kidney,expr_liver

# 2) Train
python fish_amp_cond_diffusion.py train \
  --csv data/fish_amp_clean.csv \
  --ckpt ckpt/diff.pt \
  --epochs 20 --batch 64 --lr 3e-4 \
  --d-model 384 --n-layers 6 --heads 6 --dropout 0.1 \
  --t-steps 32 --mask-schedule cosine --cfg-uncond-prob 0.15

# 3) Sample (expression-guided)
python fish_amp_cond_diffusion.py sample \
  --ckpt ckpt/diff.pt \
  --n 128 --len-min 20 --len-max 35 \
  --temperature 1.0 --topk 10 --cfg-scale 2.0 \
  --expr-json '{"expr_gill": 1.2, "expr_liver": -0.3, "expr_kidney": 0.4}' \
  --expr-cols expr_gill,expr_liver,expr_kidney \
  --out gen/peptides.csv

Notes
-----
- Expression features are treated as **continuous conditions** via an MLP → FiLM modulation of each block.
- Classifier-Free Guidance (CFG) is implemented by randomly dropping conditions during training.
- The sampler progressively unmasks positions (MaskGIT-like). Top-k & temperature control diversity.
- You can finetune from a pretrained ckpt by adding --resume ckpt/pretrain.pt.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------
# Tokenizer utilities
# ------------------------
AA_STANDARD = list("ACDEFGHIKLMNPQRSTVWY")
SPECIALS = ["X"]  # map uncommon letters to X
PAD, BOS, EOS, MASK = "<PAD>", "<BOS>", "<EOS>", "<MASK>"
VOCAB = [PAD, BOS, EOS, MASK] + AA_STANDARD + SPECIALS
stoi = {t: i for i, t in enumerate(VOCAB)}
itos = {i: t for t, i in stoi.items()}

PAD_ID = stoi[PAD]
BOS_ID = stoi[BOS]
EOS_ID = stoi[EOS]
MASK_ID = stoi[MASK]

UNSEEN_MAP = {"B": "X", "Z": "X", "J": "X", "U": "X", "O": "X"}


def sanitize_seq(seq: str) -> str:
    s = (seq or "").strip().upper()
    out = []
    for ch in s:
        if ch in stoi:
            out.append(ch)
        elif ch in UNSEEN_MAP:
            out.append(UNSEEN_MAP[ch])
        elif ch in AA_STANDARD:
            out.append(ch)
        else:
            # skip non-amino characters
            continue
    return "".join(out)


def encode(seq: str, max_len: int) -> List[int]:
    seq = sanitize_seq(seq)
    seq = seq[: max(0, max_len - 2)]  # reserve for BOS, EOS
    toks = [BOS_ID] + [stoi.get(ch, stoi["X"]) for ch in seq] + [EOS_ID]
    if len(toks) < max_len:
        toks = toks + [PAD_ID] * (max_len - len(toks))
    else:
        toks = toks[:max_len]
        toks[-1] = EOS_ID  # ensure EOS present
    return toks


def decode(toks: List[int]) -> str:
    chars = []
    for idx in toks:
        if idx in (PAD_ID, BOS_ID):
            continue
        if idx == EOS_ID:
            break
        if idx == MASK_ID:
            # treat mask as X when decoding
            chars.append("X")
        else:
            chars.append(itos.get(idx, "X"))
    return "".join(chars)

# ------------------------
# Expression utilities
# ------------------------

def standardize_expr(df: pd.DataFrame, expr_cols: List[str]) -> pd.DataFrame:
    # If columns look like raw counts/TPM/FPKM, apply log1p; then per-col z-score
    X = df[expr_cols].copy()
    # Heuristic: if max > 50, do log1p
    if np.nanmax(X.to_numpy()) > 50:
        X = np.log1p(X)
    # z-score per column
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
    df[expr_cols] = X
    return df


# ------------------------
# Dataset
# ------------------------
@dataclass
class TrainConfig:
    max_len: int = 60
    t_steps: int = 32
    mask_schedule: str = "cosine"  # or linear
    cfg_uncond_prob: float = 0.15
    length_bins: Tuple[Tuple[int, int], ...] = ((8, 15), (16, 25), (26, 35), (36, 60))


class PeptideDataset(Dataset):
    def __init__(self, df: pd.DataFrame, expr_cols: List[str], cfg: TrainConfig):
        self.df = df.reset_index(drop=True)
        self.expr_cols = expr_cols
        self.cfg = cfg
        self.max_len = cfg.max_len

        # Precompute encoded sequences
        self.enc = [torch.tensor(encode(s, self.max_len), dtype=torch.long)
                    for s in self.df["sequence"].tolist()]

        # Discrete side labels (optional)
        self.active = torch.tensor(self.df.get("active", pd.Series([np.nan]*len(df))).fillna(-1).astype(int).tolist())
        self.hemo = torch.tensor(self.df.get("hemolytic", pd.Series([np.nan]*len(df))).fillna(-1).astype(int).tolist())

        # Expression matrix (standardized already)
        self.expr = torch.tensor(self.df[self.expr_cols].to_numpy(), dtype=torch.float32) if self.expr_cols else None

        # Length bins
        self.len_bins = torch.tensor([self._len_bin(len(sanitize_seq(s))) for s in self.df["sequence"]], dtype=torch.long)

    def _len_bin(self, L: int) -> int:
        for i, (a, b) in enumerate(self.cfg.length_bins):
            if a <= L <= b:
                return i
        return len(self.cfg.length_bins) - 1

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x0 = self.enc[idx]  # [L]
        L = (x0 != PAD_ID).sum().item()
        # sample time step
        t = random.randint(1, self.cfg.t_steps)
        mask_prob = self._mask_rate(t)
        x_t = x0.clone()
        # mask internal aa positions (exclude PAD/BOS/EOS)
        token_maskable = (x0 != PAD_ID) & (x0 != BOS_ID) & (x0 != EOS_ID)
        to_mask = (torch.rand_like(x0, dtype=torch.float32) < mask_prob) & token_maskable
        x_t[to_mask] = MASK_ID

        # conditions
        active = self.active[idx]
        hemo = self.hemo[idx]
        len_bin = self.len_bins[idx]
        expr = self.expr[idx] if self.expr is not None else torch.zeros(0)

        # for CFG dropout
        cond_drop = (random.random() < self.cfg.cfg_uncond_prob)

        return {
            "x0": x0,
            "x_t": x_t,
            "t": torch.tensor(t, dtype=torch.long),
            "active": active,
            "hemo": hemo,
            "len_bin": len_bin,
            "expr": expr,
            "cond_drop": torch.tensor(1 if cond_drop else 0, dtype=torch.uint8),
        }

    def _mask_rate(self, t: int) -> float:
        u = t / self.cfg.t_steps
        if self.cfg.mask_schedule == "cosine":
            # slow at first, faster later
            return math.sin(0.5 * math.pi * u) ** 2
        else:  # linear
            return u


# ------------------------
# Model
# ------------------------
class SinusoidalTime(nn.Module):
    def __init__(self, d_model: int, max_steps: int):
        super().__init__()
        self.d_model = d_model
        self.max_steps = max_steps
        # precompute table
        pe = torch.zeros(max_steps + 1, d_model)
        position = torch.arange(0, max_steps + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, t: torch.Tensor):  # [B]
        return self.pe[t]


class FiLM(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 2 * d_model),  # gamma, beta
        )

    def forward(self, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        gb = self.net(cond)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma, beta


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float, mlp_ratio: float = 4.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(int(d_model * mlp_ratio), d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # Self-attention
        x = x + self.drop(self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0])
        # MLP
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class DiffusionSeqModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, heads: int, dropout: float,
                 t_steps: int, expr_dim: int, n_len_bins: int, use_side_labels: bool = True):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(512, d_model)
        self.time = SinusoidalTime(d_model, t_steps)

        # Discrete side labels (active/hemolytic/len_bin)
        self.use_side = use_side_labels
        if use_side_labels:
            self.active_emb = nn.Embedding(3, d_model)  # -1,0,1 → shift to {0,1,2}
            self.hemo_emb = nn.Embedding(3, d_model)
            self.len_emb = nn.Embedding(n_len_bins, d_model)
        else:
            self.register_buffer("dummy", torch.zeros(1))

        # Expression encoder → cond embedding
        self.expr_mlp = nn.Sequential(
            nn.Linear(max(1, expr_dim), d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        ) if expr_dim > 0 else None

        # FiLM per block (conditioned on time + expr + side labels)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, heads, dropout) for _ in range(n_layers)])
        self.films = nn.ModuleList([FiLM(d_model, d_model * 2) for _ in range(n_layers)])
        self.ln_film = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, expr: Optional[torch.Tensor],
                active: Optional[torch.Tensor], hemo: Optional[torch.Tensor], len_bin: Optional[torch.Tensor],
                cond_drop: Optional[torch.Tensor] = None):
        """Predict x0 logits given corrupted tokens x_t and conditions.
        x_t: [B,L] int
        t:   [B]   int
        expr: [B,E] float (can be None)
        active/hemo: [B] int in {-1,0,1}
        len_bin: [B] int
        cond_drop: [B] uint8 (1 means drop conditions for CFG)
        """
        B, L = x_t.shape
        device = x_t.device

        tok = self.tok_emb(x_t)  # [B,L,D]
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        pos = self.pos_emb(pos_ids)
        h = tok + pos

        # Build conditioning vector c
        t_emb = self.time(t)  # [B,D]
        cond = t_emb

        if self.expr_mlp is not None and expr is not None and expr.numel() > 0:
            cond = cond + self.expr_mlp(expr)

        if self.use_side and active is not None and hemo is not None and len_bin is not None:
            act_shift = (active + 1).clamp(0, 2)
            hemo_shift = (hemo + 1).clamp(0, 2)
            cond = cond + self.active_emb(act_shift) + self.hemo_emb(hemo_shift) + self.len_emb(len_bin)

        if cond_drop is not None:
            # zero-out cond for dropped samples (unconditional path)
            mask = (cond_drop > 0).float().unsqueeze(-1)
            cond = cond * (1.0 - mask)

        # Apply FiLM modulation per block
        for blk, film in zip(self.blocks, self.films):
            gamma, beta = film(self.ln_film(cond))  # [B,D] each
            h = blk(h)
            # broadcast FiLM over sequence length
            h = (1.0 + gamma.unsqueeze(1)) * h + beta.unsqueeze(1)

        logits = self.head(h)  # [B,L,V]
        return logits


# ------------------------
# Training / Sampling helpers
# ------------------------

def ce_loss_x0(logits: torch.Tensor, x0: torch.Tensor, pad_id: int, mask_weight: Optional[torch.Tensor] = None):
    # logits: [B,L,V], x0: [B,L]
    B, L, V = logits.shape
    loss = F.cross_entropy(logits.view(B * L, V), x0.view(-1), reduction='none')
    loss = loss.view(B, L)
    # mask out PAD positions
    pad_mask = (x0 != pad_id).float()
    loss = loss * pad_mask
    if mask_weight is not None:
        loss = loss * mask_weight
    return loss.sum() / (pad_mask.sum() + 1e-6)


def mask_weight_from_xt(x_t: torch.Tensor, mask_id: int, w_masked: float = 1.0):
    # emphasize positions that were masked at input
    return 1.0 + (x_t == mask_id).float() * w_masked


@torch.no_grad()
def progressive_unmask_sampler(model: DiffusionSeqModel, n: int, len_min: int, len_max: int, t_steps: int,
                               temperature: float, topk: int, cfg_scale: float,
                               expr_vec: Optional[torch.Tensor], device: torch.device):
    model.eval()
    L = len_max  # generate at max length; users can trim by EOS
    x = torch.full((n, L), MASK_ID, dtype=torch.long, device=device)
    # set BOS/EOS placeholders
    x[:, 0] = BOS_ID
    x[:, -1] = EOS_ID

    # side labels: unknown by default
    active = torch.full((n,), -1, dtype=torch.long, device=device)
    hemo = torch.full((n,), -1, dtype=torch.long, device=device)

    # naive length bin estimate by target length (use len_max)
    def len_bin_of(Li: int):
        bins = ((8, 15), (16, 25), (26, 35), (36, 60))
        for i, (a, b) in enumerate(bins):
            if a <= Li <= b:
                return i
        return len(bins) - 1

    len_bin = torch.tensor([len_bin_of((len_min + len_max)//2)] * n, dtype=torch.long, device=device)

    remain = (x == MASK_ID) & (x != BOS_ID) & (x != EOS_ID)
    for step in range(1, t_steps + 1):
        t = torch.full((n,), step, dtype=torch.long, device=device)
        # CFG: need conditional and unconditional logits
        logits_cond = model(x, t, expr_vec, active, hemo, len_bin, cond_drop=torch.zeros(n, dtype=torch.uint8, device=device))
        logits_uncond = model(x, t, expr_vec, active, hemo, len_bin, cond_drop=torch.ones(n, dtype=torch.uint8, device=device))
        logits = (1 + cfg_scale) * logits_cond - cfg_scale * logits_uncond

        # focus only on remaining masked positions
        logits = logits / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)

        # don't allow PAD/BOS as samples; allow EOS
        probs[:, :, PAD_ID] = 0.0
        probs[:, :, BOS_ID] = 0.0

        if topk > 0:
            topk_vals, topk_idx = torch.topk(probs, k=min(topk, probs.size(-1)), dim=-1)
            new_probs = torch.zeros_like(probs)
            new_probs.scatter_(dim=-1, index=topk_idx, src=topk_vals)
            probs = new_probs / (new_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # sample tokens only at remaining positions
        B, L_, V = probs.shape
        flat_probs = probs[remain]
        if flat_probs.numel() > 0:
            sampled = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
            x[remain] = sampled

        # update remaining set: progressively unmask a fraction based on confidence
        with torch.no_grad():
            conf, _ = torch.max(probs, dim=-1)  # [B,L]
            conf = conf.masked_fill(~remain, -1.0)
            # unmask top q fraction of remaining per step
            q = step / t_steps  # increasing fraction
            num_remain = remain.float().sum(dim=1)  # [B]
            take = (num_remain * q).long().clamp(min=1)
            for i in range(n):
                if num_remain[i] <= 0:
                    continue
                vals, idxs = torch.topk(conf[i], k=min(int(take[i].item()), int(num_remain[i].item())))
                remain[i, idxs] = False

        if remain.float().sum() == 0:
            break

    return x


# ------------------------
# Prep logic
# ------------------------

def cmd_prep(args):
    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    df = pd.read_csv(args.csv)

    # optional merge expression matrix by key
    expr_cols = None
    if args.expr_table:
        expr = pd.read_csv(args.expr_table)
        if args.csv_key not in df.columns or args.expr_key not in expr.columns:
            raise ValueError("csv-key / expr-key not found in columns")
        df = df.merge(expr, left_on=args.csv_key, right_on=args.expr_key, how='left')

    # sanitize sequence
    df['sequence'] = df['sequence'].astype(str).map(sanitize_seq)
    df = df[df['sequence'].str.len() > 0]
    df = df[(df['sequence'].str.len() >= args.min_len) & (df['sequence'].str.len() <= args.max_len)]
    df = df.drop_duplicates(subset=['sequence'])

    # choose expression columns
    if args.expr_cols:
        expr_cols = [c.strip() for c in args.expr_cols.split(',') if c.strip() in df.columns]
    else:
        # auto-detect
        expr_cols = [c for c in df.columns if c.startswith('expr_') or c.startswith('tpm_') or c.startswith('fpkm_')]

    if expr_cols:
        df = standardize_expr(df, expr_cols)

    df.to_csv(args.out, index=False)
    print(f"[prep] saved cleaned CSV to {args.out} (N={len(df)}), expr_cols={expr_cols}")


# ------------------------
# Train logic
# ------------------------

def cmd_train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    df = pd.read_csv(args.csv)
    if args.expr_cols:
        expr_cols = [c.strip() for c in args.expr_cols.split(',') if c.strip() in df.columns]
    else:
        expr_cols = [c for c in df.columns if c.startswith('expr_') or c.startswith('tpm_') or c.startswith('fpkm_')]

    cfg = TrainConfig(max_len=args.max_len, t_steps=args.t_steps,
                      mask_schedule=args.mask_schedule, cfg_uncond_prob=args.cfg_uncond_prob)

    ds = PeptideDataset(df, expr_cols, cfg)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)

    model = DiffusionSeqModel(
        vocab_size=len(VOCAB), d_model=args.d_model, n_layers=args.n_layers, heads=args.heads,
        dropout=args.dropout, t_steps=args.t_steps,
        expr_dim=len(expr_cols), n_len_bins=len(cfg.length_bins), use_side_labels=True
    ).to(device)

    if args.resume and os.path.isfile(args.resume):
        print(f"[train] loading from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda') and args.amp)

    os.makedirs(os.path.dirname(args.ckpt), exist_ok=True) if os.path.dirname(args.ckpt) else None

    step = 0
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in dl:
            step += 1
            x0 = batch['x0'].to(device)
            x_t = batch['x_t'].to(device)
            t = batch['t'].to(device)
            active = batch['active'].to(device)
            hemo = batch['hemo'].to(device)
            len_bin = batch['len_bin'].to(device)
            expr = batch['expr'].to(device) if ds.expr is not None else None
            cond_drop = batch['cond_drop'].to(device)

            mw = mask_weight_from_xt(x_t, MASK_ID, w_masked=args.mask_weight)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda') and args.amp):
                logits = model(x_t, t, expr, active, hemo, len_bin, cond_drop)
                loss = ce_loss_x0(logits, x0, PAD_ID, mask_weight=mw)
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            if step % args.log_every == 0:
                print(f"[epoch {epoch}] step {step} loss={loss.item():.4f}")

        # save per epoch
        torch.save(model.state_dict(), args.ckpt)
        print(f"[train] saved ckpt to {args.ckpt}")
        if loss.item() < best:
            best = loss.item()
            if args.ckpt_best:
                torch.save(model.state_dict(), args.ckpt_best)
                print(f"[train] new best {best:.4f} saved to {args.ckpt_best}")


# ------------------------
# Sample logic
# ------------------------

def cmd_sample(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Build expr vector order
    if args.expr_cols:
        expr_cols = [c.strip() for c in args.expr_cols.split(',') if c.strip()]
    else:
        expr_cols = []

    if args.expr_json:
        d = json.loads(args.expr_json)
        vec = [float(d.get(c, 0.0)) for c in expr_cols]
        expr = torch.tensor(vec, dtype=torch.float32, device=device).unsqueeze(0)
        expr = expr.repeat(args.n, 1)
    else:
        expr = torch.zeros((args.n, max(1, len(expr_cols))), dtype=torch.float32, device=device)
        if len(expr_cols) == 0:
            expr = None  # no expression

    model = DiffusionSeqModel(
        vocab_size=len(VOCAB), d_model=args.d_model, n_layers=args.n_layers, heads=args.heads,
        dropout=args.dropout, t_steps=args.t_steps,
        expr_dim=(0 if expr is None else expr.shape[1]), n_len_bins=4, use_side_labels=True
    ).to(device)

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"ckpt not found: {args.ckpt}")
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    x = progressive_unmask_sampler(
        model, n=args.n, len_min=args.len_min, len_max=args.len_max, t_steps=args.t_steps,
        temperature=args.temperature, topk=args.topk, cfg_scale=args.cfg_scale,
        expr_vec=expr, device=device
    )

    seqs = [decode(x[i].tolist()) for i in range(args.n)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True) if os.path.dirname(args.out) else None
    out_rows = []
    for s in seqs:
        row = {"sequence": s}
        if expr is not None:
            for i, c in enumerate(expr_cols):
                row[c] = float(expr[0, i].item())
        out_rows.append(row)
    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    print(f"[sample] wrote {len(seqs)} sequences → {args.out}")


# ------------------------
# CLI
# ------------------------

def build_parser():
    p = argparse.ArgumentParser("Fish AMP Conditional Discrete Diffusion")
    sub = p.add_subparsers(dest='cmd', required=True)

    sp = sub.add_parser('prep', help='clean CSV & merge/standardize expression features')
    sp.add_argument('--csv', required=True)
    sp.add_argument('--out', required=True)
    sp.add_argument('--min-len', type=int, default=8)
    sp.add_argument('--max-len', type=int, default=60)
    sp.add_argument('--expr-table', default=None, help='optional external expression CSV')
    sp.add_argument('--csv-key', default='gene_id')
    sp.add_argument('--expr-key', default='gene_id')
    sp.add_argument('--expr-cols', default=None, help='comma-separated expression columns to keep')
    sp.set_defaults(func=cmd_prep)

    st = sub.add_parser('train', help='train diffusion model')
    st.add_argument('--csv', required=True)
    st.add_argument('--expr-cols', default=None)
    st.add_argument('--ckpt', required=True)
    st.add_argument('--ckpt-best', default=None)
    st.add_argument('--resume', default=None)
    st.add_argument('--epochs', type=int, default=20)
    st.add_argument('--batch', type=int, default=64)
    st.add_argument('--lr', type=float, default=3e-4)
    st.add_argument('--weight-decay', type=float, default=0.01)
    st.add_argument('--d-model', type=int, default=384)
    st.add_argument('--n-layers', type=int, default=6)
    st.add_argument('--heads', type=int, default=6)
    st.add_argument('--dropout', type=float, default=0.1)
    st.add_argument('--max-len', type=int, default=60)
    st.add_argument('--t-steps', type=int, default=32)
    st.add_argument('--mask-schedule', choices=['cosine','linear'], default='cosine')
    st.add_argument('--cfg-uncond-prob', type=float, default=0.15)
    st.add_argument('--mask-weight', type=float, default=1.0)
    st.add_argument('--grad-clip', type=float, default=1.0)
    st.add_argument('--amp', action='store_true')
    st.add_argument('--log-every', type=int, default=100)
    st.set_defaults(func=cmd_train)

    ss = sub.add_parser('sample', help='sample peptides')
    ss.add_argument('--ckpt', required=True)
    ss.add_argument('--out', required=True)
    ss.add_argument('--n', type=int, default=128)
    ss.add_argument('--len-min', type=int, default=20)
    ss.add_argument('--len-max', type=int, default=35)
    ss.add_argument('--t-steps', type=int, default=32)
    ss.add_argument('--temperature', type=float, default=1.0)
    ss.add_argument('--topk', type=int, default=10)
    ss.add_argument('--cfg-scale', type=float, default=2.0)
    ss.add_argument('--expr-json', default=None, help='JSON dict of expr values matching --expr-cols order')
    ss.add_argument('--expr-cols', default=None, help='comma-separated expression column names (order matters)')
    ss.add_argument('--d-model', type=int, default=384)
    ss.add_argument('--n-layers', type=int, default=6)
    ss.add_argument('--heads', type=int, default=6)
    ss.add_argument('--dropout', type=float, default=0.0)
    ss.set_defaults(func=cmd_sample)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
