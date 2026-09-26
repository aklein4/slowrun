"""
Loss-controlled Gaussian matrix noise on the gated limited-compute Slowrun model.
Based on qlabs-eng/slowrun@52e7441f862c3295c0f5695933438dac78f7fc5b.

Single-file submission. The original model layout and Muon update are retained;
new code supplies learned gains, post-step normalization, noise and ensembles.

Usage:
    python research/loss_budget/train.py --output local_data/runs/loss-budget
"""

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

MAX_SEQ_LEN, DEPTH, N_HEAD, N_EMBD = 2048, 30, 14, 1792
WINDOW_PATTERN = "SSSL"
print0 = print


def git_info():
    """Provenance when available; the submission also runs outside a checkout."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ))
        return {"git_commit": commit, "git_dirty": dirty}
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


# =============================================================================
# CLI arguments
# =============================================================================

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-data", default="fineweb_data/fineweb_train.pt")
    p.add_argument("--val-data", default="fineweb_data/fineweb_val.pt")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-epochs", type=int, default=40)
    p.add_argument(
        "--steps", type=int, help="Override planned horizon for bounded experiments"
    )
    p.add_argument(
        "--stop-after",
        type=int,
        help="Pause at this absolute step, preserving the planned horizon",
    )
    p.add_argument("--device-batch-size", type=int, default=1)
    p.add_argument(
        "--total-batch-size", type=int, default=524288, help="Tokens per optimizer step"
    )
    p.add_argument("--sequence-len", type=int, default=2048)
    p.add_argument("--n-layer", type=int, default=30)
    p.add_argument("--n-head", type=int, default=14)
    p.add_argument("--n-embd", type=int, default=1792)
    p.add_argument("--matrix-lr", type=float, default=0.02)
    p.add_argument("--scale-lr", type=float, default=0.01)
    p.add_argument(
        "--scalar-lr",
        type=float,
        default=0.125,
        help="Original effective scalar LR; residual/skip use 0.01 times this",
    )
    p.add_argument("--embedding-lr", type=float, default=0.002)
    p.add_argument("--unembedding-lr", type=float, default=0.02)
    p.add_argument("--scale-weight-decay", type=float, default=0.0)
    p.add_argument("--lr-warmup-steps", type=int, default=100)
    p.add_argument(
        "--final-lr-fraction",
        type=float,
        default=1.0,
        help="1 keeps constant sampling LR; <1 uses linear decay after warmup",
    )
    p.add_argument(
        "--loss-target",
        type=float,
        default=3.5,
        help="Training CE in nats/token; a feedback target, not a hard bound",
    )
    p.add_argument("--loss-ema-beta", type=float, default=0.98)
    p.add_argument(
        "--noise-seed-std",
        type=float,
        default=1e-4,
        help="Relative tangent RMS at base LR when activated",
    )
    p.add_argument("--noise-max-std", type=float, default=0.5)
    p.add_argument("--noise-growth", type=float, default=1.02)
    p.add_argument("--noise-interval", type=int, default=10)
    p.add_argument("--noise-warmup-steps", type=int, default=100)
    p.add_argument("--loss-tolerance", type=float, default=0.01)
    p.add_argument("--no-noise", action="store_true")
    p.add_argument(
        "--snapshots", type=int, default=10, help="0 disables snapshots for ablations"
    )
    p.add_argument(
        "--eval-batches",
        type=int,
        help="Bound evaluation; default uses the finite full split once",
    )
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Resumable checkpoint interval; final/pause checkpoint always saved",
    )
    p.add_argument("--skip-ensemble", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate the manifest ensemble without training",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Compile forward and Muon polar iteration",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p


def lr_multiplier(step, total, warmup, final):
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return 1 - min(1, max(0, progress)) * (1 - final)


def validate_args(args):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "This trainer is single-GPU; launch with python, not multi-rank torchrun"
        )
    if (
        min(
            args.num_epochs,
            args.device_batch_size,
            args.sequence_len,
            args.n_layer,
            args.n_head,
            args.n_embd,
        )
        < 1
    ):
        raise ValueError("Model, batch and epoch dimensions must be positive")
    if args.n_embd % args.n_head or (args.n_embd // args.n_head) % 2:
        raise ValueError(
            "Width must be divisible by heads and head dimension must be even"
        )
    if args.n_embd < 32:
        raise ValueError("The original value gate requires at least 32 embedding channels")
    if args.total_batch_size < 1 or args.total_batch_size % (
        args.device_batch_size * args.sequence_len
    ):
        raise ValueError("Total batch tokens must be divisible by device batch tokens")
    if not 0 <= args.final_lr_fraction <= 1:
        raise ValueError("Invalid final LR fraction")
    if any(
        not math.isfinite(x) or x <= 0
        for x in (
            args.matrix_lr,
            args.scale_lr,
            args.scalar_lr,
            args.embedding_lr,
            args.unembedding_lr,
        )
    ):
        raise ValueError("Learning rates must be finite and positive")
    if args.scale_weight_decay < 0 or not math.isfinite(args.scale_weight_decay):
        raise ValueError("Scale decay must be finite and nonnegative")
    if min(args.snapshots, args.lr_warmup_steps, args.eval_every, args.save_every) < 0:
        raise ValueError("Counts and intervals must be nonnegative")
    if any(
        v is not None and v < 1
        for v in (args.steps, args.stop_after, args.eval_batches)
    ):
        raise ValueError("Explicit run and evaluation bounds must be positive")



# =============================================================================
# Utilities and persistence
# =============================================================================

def snapshot_steps(total_steps, count=10):
    if total_steps < 1 or count < 1:
        raise ValueError("Training steps and snapshot count must be positive")
    first = (total_steps + 1) // 2
    if count > total_steps - first + 1:
        raise ValueError(
            "Not enough distinct steps in the second half for the requested snapshots"
        )
    if count == 1:
        return [total_steps]
    return [
        first + round(i * (total_steps - first) / (count - 1)) for i in range(count)
    ]


def atomic_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def autocast(device):
    return (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def load_snapshot(model, path):
    checkpoint = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
    if (
        "model_config" in checkpoint
        and asdict(model.config) != checkpoint["model_config"]
    ):
        raise ValueError("Snapshot architecture differs from the prediction model")
    model.load_state_dict(checkpoint["model"], strict=True)
    return checkpoint


def cpu_state(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def save_training(
    path, model, muon, adam, controller, noise_generator, step, micro_seen, contract
):
    atomic_save(
        {
            "version": 1,
            "model": cpu_state(model),
            "muon": muon.state_dict(),
            "adam": adam.state_dict(),
            "controller": controller.state_dict(),
            "noise_rng": noise_generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all()
                if next(model.parameters()).is_cuda
                else []
            ),
            "step": step,
            "micro_seen": micro_seen,
            "contract": contract,
        },
        path,
    )


@torch.no_grad()
def sphere_errors(model):
    errors = {"matrix": [], "embedding": []}
    for module in model.modules():
        if isinstance(module, ScaledLinear):
            if module.row_normalized:
                error = (module.weight.norm(dim=1) - 1).abs().max()
                errors["embedding"].append(error)
            else:
                error = (
                    module.weight.norm() / math.sqrt(module.weight.shape[0]) - 1
                ).abs()
                errors["matrix"].append(error)
    return {key: torch.stack(values).max().item() for key, values in errors.items()}



# =============================================================================
# Flash Attention (FA2 on CUDA; SDPA for CPU checks)
# =============================================================================

def attention(q, k, v, causal=True, window_size=(-1, -1)):
    # FA2 is locally available on the test GH200; SDPA keeps CPU tests portable.
    if q.is_cuda:
        from flash_attn import flash_attn_func

        return flash_attn_func(q, k, v, causal=causal, window_size=window_size)
    length = q.shape[1]
    pos = torch.arange(length, device=q.device)
    delta = pos[:, None] - pos[None, :]
    mask = delta >= 0
    if window_size[0] >= 0:
        mask &= delta <= window_size[0]
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask
    )
    return out.transpose(1, 2)


flash_attn = SimpleNamespace(flash_attn_func=attention)


# =============================================================================
# Post-step direction and magnitude parameterization
# =============================================================================

class ScaledLinear(nn.Module):
    """W = diag(row_scale) direction diag(col_scale), projected after each step.

    Matrix radius sqrt(out_features) preserves fan-in initialization. Output
    embeddings instead have unit row norms. Signed gains permit exact zero
    initialization of residual projections/gates without a zero direction.
    """

    def __init__(self, in_features, out_features, bias=False, row_normalized=False):
        super().__init__()
        if bias:
            raise ValueError("The gated baseline has no linear biases")
        self.row_normalized = row_normalized
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.row_scale = nn.Parameter(torch.ones(out_features))
        self.col_scale = nn.Parameter(torch.ones(in_features))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.normal_(self.weight)
        self.project_()
        self.row_scale.fill_(1.0)
        self.col_scale.fill_(1.0)

    @torch.no_grad()
    def project_(self):
        if self.row_normalized:
            self.weight.div_(self.weight.norm(dim=1, keepdim=True).clamp_min(1e-12))
        else:
            self.weight.mul_(
                math.sqrt(self.weight.shape[0]) / self.weight.norm().clamp_min(1e-12)
            )

    def forward(self, x):
        y = F.linear(x * self.col_scale.to(x.dtype), self.weight)
        return y * self.row_scale.to(y.dtype)


class ScaledEmbedding(ScaledLinear):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__(embedding_dim, num_embeddings, row_normalized=True)

    def forward(self, idx):
        x = F.embedding(idx, self.weight)
        return x * F.embedding(idx, self.row_scale[:, None]) * self.col_scale


@torch.no_grad()
def parameterize_model_(model):
    """Replace initialized baseline modules without changing their effective weights.

    Nonzero matrices retain the baseline draw exactly up to roundoff. Only zero
    matrices need an auxiliary random direction, multiplied by zero row gains.
    """
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, ScaledLinear):
            continue
        if isinstance(module, (nn.Linear, nn.Embedding)):
            replacements.append((name, module))
    for name, module in replacements:
        original = module.weight.detach()
        rows, cols = original.shape
        embedding = isinstance(module, nn.Embedding)
        row_normalized = embedding or name == "lm_head"
        # The constructor is only a shape allocation; do not consume extra RNG
        # for nonzero baseline weights. The zero matrices use new directions.
        with torch.device("meta"):
            scaled = (
                ScaledEmbedding(rows, cols)
                if embedding
                else ScaledLinear(cols, rows, row_normalized=row_normalized)
            )
        scaled.to_empty(device=original.device)
        scaled.col_scale.fill_(1.0)
        if row_normalized:
            norms = original.norm(dim=1)
            if (norms == 0).any():
                raise ValueError("Baseline embedding rows must be nonzero")
            scaled.weight.copy_(original / norms[:, None])
            scaled.row_scale.copy_(norms)
        else:
            norm = original.norm()
            radius = math.sqrt(rows)
            if norm == 0:
                nn.init.normal_(scaled.weight)
                scaled.project_()
                scaled.row_scale.zero_()
            else:
                scaled.weight.copy_(original * (radius / norm))
                scaled.row_scale.fill_(norm / radius)
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, scaled)

# =============================================================================
# GPT Model
# =============================================================================

# BEGIN BASELINE MODEL
@dataclass
class GPTConfig:
    sequence_len: int = MAX_SEQ_LEN
    vocab_size: int = 50257
    n_layer: int = DEPTH
    n_head: int = N_HEAD
    n_kv_head: int = N_HEAD
    n_embd: int = N_EMBD
    window_pattern: str = WINDOW_PATTERN

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

def has_ve(layer_idx, n_layer):
    """Value Embedding on alternating layers, last layer always included."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.resid_dropout = nn.Identity()
        self.ve_gate_channels = 32
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None
        # Attention gate: per-head gating to enable context-based no-op
        self.attn_gate_channels = 12
        self.attn_gate = nn.Linear(self.attn_gate_channels, self.n_head, bias=False)

    def forward(self, x, ve, cos_sin, window_size):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        # Value residual (ResFormer)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        # Attention gate: per-head sigmoid gate
        y = y * torch.sigmoid(self.attn_gate(x[..., :self.attn_gate_channels])).unsqueeze(-1)
        y = y.contiguous().view(B, T, -1)
        return self.resid_dropout(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden = 256 * ((8 * config.n_embd // 3 + 255) // 256)
        self.c_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_fc = nn.Linear(config.n_embd, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, config.n_embd, bias=False)
        self.resid_dropout = nn.Identity()

    def forward(self, x):
        return self.resid_dropout(self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size):
        x = x + self.attn(norm(x), ve, cos_sin, window_size)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        super().__init__()
        self.config = config
        self.window_sizes = self._compute_window_sizes(config)
        padded_vocab = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab}")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab, config.n_embd),
            "h": nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, padded_vocab, bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.ve_projs = nn.ModuleDict({str(i): nn.Linear(config.n_embd, kv_dim, bias=False) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # U-Net skip connections: encoder layer i → decoder layer (n_layer - 1 - i)
        self.encoder_layers = config.n_layer // 2
        self.skip_weights = nn.Parameter(torch.ones(self.encoder_layers))
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        s = 3**0.5 * self.config.n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_gate.weight, -s, s)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        for proj in self.ve_projs.values():
            torch.nn.init.uniform_(proj.weight, -s, s)
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.zeros_(block.attn.ve_gate.weight)
            torch.nn.init.zeros_(block.attn.attn_gate.weight)
        self.skip_weights.fill_(1.0)
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Keep fp32 master directions; autocast supplies bf16 matrix operations.
        parameterize_model_(self)

    def _precompute_rotary(self, seq_len, head_dim, base=10000):
        device = self.transformer.wte.weight.device
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos().bfloat16(), freqs.sin().bfloat16()
        return cos[None, :, None, :], sin[None, :, None, :]

    def _compute_window_sizes(self, config):
        pattern = config.window_pattern.upper()
        long_w, short_w = config.sequence_len, config.sequence_len // 2
        char_to_w = {"L": (long_w, 0), "S": (short_w, 0)}
        sizes = [char_to_w[pattern[i % len(pattern)]] for i in range(config.n_layer)]
        sizes[-1] = (long_w, 0)  # final layer always full context
        return sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        nparams = sum(p.numel() for p in self.parameters())
        ve_numel = sum(p.weight.numel() for p in self.ve_projs.values())
        nparams_exclude = self.transformer.wte.weight.numel() + ve_numel + self.resid_lambdas.numel() + self.x0_lambdas.numel()
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        attn_flops = sum(12 * h * q * min(w[0], t) if w[0] >= 0 else 12 * h * q * t for w in self.window_sizes)
        return 6 * (nparams - nparams_exclude) + attn_flops

    def setup_optimizer(self, **kwargs):
        return make_optimizers(self, **kwargs)

    def forward(self, idx, targets=None, loss_reduction='mean'):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = norm(self.transformer.wte(idx))
        x0 = x
        skip_connections = []
        for i, block in enumerate(self.transformer.h):
            if i >= self.encoder_layers and skip_connections:
                skip = skip_connections.pop()
                x = x + self.skip_weights[i - self.encoder_layers] * skip
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x = block(x, ve, cos_sin, self.window_sizes[i])
            if i < self.encoder_layers:
                skip_connections.append(x)
        x = norm(x)
        logits = self.lm_head(x)[..., :self.config.vocab_size].float()
        logits = 15 * torch.tanh(logits / 15)  # softcap
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
        return logits

# END BASELINE MODEL

# =============================================================================
# Optimizer: MuonAdamW (Muon for matrices, AdamW for embeddings/scalars)
# =============================================================================

# BEGIN BASELINE MUON
# Polar Express coefficients for orthogonalization
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,
                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)
    # Polar Express orthogonalization
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            X = a * X + X @ (b * A + c * (A @ A))
    else:
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
    g = X
    # Variance reduction
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)
    # Cautious weight decay + update
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# END BASELINE MUON
@torch.no_grad()
def tangent_noise_(p, relative_std, generator):
    """Expected squared tangent displacement = relative_std**2 * ||p||**2."""
    if relative_std == 0:
        return
    noise = torch.randn(p.shape, device=p.device, dtype=p.dtype, generator=generator)
    norm_sq = p.square().sum().clamp_min(1e-24)
    noise.sub_(p * ((noise * p).sum() / norm_sq))
    p.add_(noise * (relative_std * (norm_sq / (p.numel() - 1)).sqrt()))


class SphereMuon(torch.optim.Optimizer):
    """Muon with the record's update algebra, then noise and retraction.

    LR keeps the record's aspect-ratio scaling. Noise is an angular RMS at the
    reference LR; sqrt(lr/reference_lr) gives a Langevin-style diffusion scaling.
    No claim of exact posterior sampling is implied by this update.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, beta2=0.95, compile=False):
        super().__init__(
            params,
            dict(
                lr=lr, initial_lr=lr, momentum=momentum, beta2=beta2, weight_decay=0.0
            ),
        )
        self.update = (
            torch.compile(muon_step_fused, fullgraph=True)
            if compile
            else muon_step_fused
        )

    @torch.no_grad()
    def step(self, noise_std=0.0, generator=None):
        if noise_std < 0 or not math.isfinite(noise_std):
            raise ValueError("noise_std must be finite and nonnegative")
        for group in self.param_groups:
            lr = group["lr"]
            if lr == 0:
                continue
            for p in group["params"]:
                if p.ndim != 2 or p.numel() < 2:
                    raise ValueError("Muon requires nontrivial matrices")
                if p.grad is None:
                    continue
                state = self.state[p]
                axis = 1 if p.shape[0] >= p.shape[1] else 0
                if not state:
                    state["momentum"] = torch.zeros_like(p)
                    state["variance"] = torch.zeros_like(p.mean(dim=axis, keepdim=True))
                # Same shaped update and LR adjustment as _compute_muon in the
                # record, with one matrix per call on this single-GPU adapter.
                self.update(
                    p.grad.unsqueeze(0),
                    p.unsqueeze(0),
                    state["momentum"].unsqueeze(0),
                    state["variance"].unsqueeze(0),
                    torch.tensor(group["momentum"]),
                    torch.tensor(lr * max(1.0, p.shape[0] / p.shape[1]) ** 0.5),
                    torch.tensor(0.0),
                    torch.tensor(group["beta2"]),
                    5,
                    -1 if axis == 1 else -2,
                )
                radius = math.sqrt(p.shape[0])
                # Re-establish the sphere before projecting the Gaussian tangent.
                p.mul_(radius / p.norm().clamp_min(1e-12))
                tangent_noise_(
                    p, noise_std * math.sqrt(lr / group["initial_lr"]), generator
                )
                p.mul_(radius / p.norm().clamp_min(1e-12))


def make_optimizers(
    model,
    matrix_lr=0.02,
    scale_lr=0.01,
    embedding_lr=0.002,
    scale_weight_decay=0.0,
    compile=False,
    unembedding_lr=0.02,
    scalar_lr=0.125,
):
    matrices, embeddings, unembeddings, scales = [], [], [], []
    for module in model.modules():
        if isinstance(module, ScaledLinear):
            if module is model.lm_head:
                unembeddings.append(module.weight)
            else:
                (embeddings if module.row_normalized else matrices).append(
                    module.weight
                )
            scales.extend([module.row_scale, module.col_scale])
    owned = {id(p) for p in matrices + embeddings + unembeddings + scales}
    scalars = [p for p in model.parameters() if id(p) not in owned]
    if {id(p) for p in scalars} != {
        id(model.resid_lambdas),
        id(model.x0_lambdas),
        id(model.skip_weights),
    }:
        raise ValueError(
            "Unrecognized scalar parameters need an explicit optimizer group"
        )
    all_params = matrices + embeddings + unembeddings + scales + scalars
    if len({id(p) for p in all_params}) != len(all_params):
        raise ValueError("Optimizer groups overlap")
    muon = SphereMuon(matrices, lr=matrix_lr, compile=compile)
    adam = torch.optim.AdamW(
        [
            dict(
                params=embeddings, lr=embedding_lr, weight_decay=0.0, role="embedding"
            ),
            dict(
                params=unembeddings,
                lr=unembedding_lr,
                weight_decay=0.0,
                role="unembedding",
            ),
            dict(
                params=scales,
                lr=scale_lr,
                weight_decay=scale_weight_decay,
                role="scale",
            ),
            # Preserve the gated record's residual/skip LRs and x0 momentum.
            dict(
                params=[model.resid_lambdas, model.skip_weights],
                lr=scalar_lr * 0.01,
                weight_decay=0.0,
                role="scalar",
            ),
            dict(
                params=[model.x0_lambdas],
                lr=scalar_lr,
                betas=(0.96, 0.95),
                weight_decay=0.0,
                role="scalar",
            ),
        ],
        betas=(0.8, 0.95),
        eps=1e-10,
        foreach=True,
    )
    for group in adam.param_groups:
        group["initial_lr"] = group["lr"]
    return muon, adam


@torch.no_grad()
def project_embeddings_(model):
    for module in model.modules():
        if isinstance(module, ScaledLinear) and module.row_normalized:
            module.project_()


@dataclass
class LossBudgetController:
    target: float = 3.5
    beta: float = 0.98
    seed_std: float = 1e-4
    max_std: float = 0.5
    growth: float = 1.02
    interval: int = 10
    warmup: int = 100
    tolerance: float = 0.01
    enabled: bool = True
    ema_sum: float = 0.0
    steps: int = 0
    noise_std: float = 0.0

    def __post_init__(self):
        values = (
            self.target,
            self.beta,
            self.seed_std,
            self.max_std,
            self.growth,
            self.tolerance,
        )
        if not all(math.isfinite(x) for x in values):
            raise ValueError("Controller settings must be finite")
        if not (
            self.target > 0
            and 0 <= self.beta < 1
            and 0 < self.seed_std <= self.max_std
            and self.growth > 1
            and self.interval > 0
            and self.warmup >= 0
            and self.tolerance >= 0
        ):
            raise ValueError("Invalid controller settings")

    @property
    def ema(self):
        return self.ema_sum / (1 - self.beta**self.steps) if self.steps else None

    def observe(self, mean_train_loss):
        if not math.isfinite(mean_train_loss):
            raise FloatingPointError("Nonfinite training loss; refusing a noise update")
        self.steps += 1
        self.ema_sum = self.beta * self.ema_sum + (1 - self.beta) * mean_train_loss
        if not self.enabled:
            self.noise_std = 0.0
        elif self.steps > self.warmup and self.steps % self.interval == 0:
            if self.ema < self.target:
                self.noise_std = min(
                    self.max_std, max(self.seed_std, self.noise_std * self.growth)
                )
            elif self.ema > self.target + self.tolerance:
                self.noise_std /= self.growth
        return self.noise_std

    def state_dict(self):
        return asdict(self)

# =============================================================================
# Dataloader: finite deterministic epochs
# =============================================================================

def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TokenBatches:
    def __init__(self, path, batch_size, sequence_len, shuffle, seed=42):
        data = torch.load(path, map_location="cpu", weights_only=True)
        size = sequence_len + 1
        if "chunks" in data:
            if data["sequence_size"] != size:
                raise ValueError("Packed sequence length does not match the model")
            self.rows = torch.cat(
                [
                    chunk.reshape(data["batch_size"], size)[:n]
                    for chunk, n in zip(data["chunks"], data["valid_counts"])
                ]
            ).long()
        elif "tokens" in data:
            tokens = data["tokens"].long()
            self.rows = tokens[: tokens.numel() // size * size].reshape(-1, size)
        else:
            raise ValueError(f"Unknown Slowrun token format: {path}")
        if self.rows.numel() == 0 or self.rows.min() < 0 or self.rows.max() >= 50257:
            raise ValueError("Expected nonempty GPT-2 tokens in [0, 50257)")
        self.batch_size = batch_size
        self.num_batches = len(self.rows) // batch_size
        if self.num_batches == 0:
            raise ValueError("Dataset is smaller than one device batch")
        self.seed, self.shuffle = seed, shuffle
        self.epoch, self.order = None, None
        self.identity = {
            "path": str(Path(path).resolve()),
            "sha256": file_sha256(path),
            "rows": len(self.rows),
            "dropped_rows": len(self.rows) % batch_size,
        }

    def batch(self, index, device):
        epoch, offset = divmod(index, self.num_batches)
        if epoch != self.epoch:
            self.order = (
                torch.randperm(
                    len(self.rows),
                    generator=torch.Generator().manual_seed(self.seed + epoch),
                )
                if self.shuffle
                else torch.arange(len(self.rows))
            )
            self.epoch = epoch
        ids = self.order[offset * self.batch_size : (offset + 1) * self.batch_size]
        rows = self.rows[ids].to(device)
        return rows[:, :-1].contiguous(), rows[:, 1:].contiguous()

# =============================================================================
# Loss evaluation and checkpoint ensembles
# =============================================================================

@torch.no_grad()
def ensemble_log_probs(model, paths, inputs):
    """Full predictive distribution with only one model resident on the GPU.

    Mutates model weights to the last member; logaddexp avoids probability
    underflow. This is a mixture of softmax distributions, not mean logits.
    """
    if not paths:
        raise ValueError("The ensemble has no snapshots")
    model.eval()
    mixture = None
    for path in paths:
        load_snapshot(model, path)
        with autocast(inputs.device):
            log_probs = F.log_softmax(model(inputs).float(), dim=-1)
        mixture = log_probs if mixture is None else torch.logaddexp(mixture, log_probs)
    return mixture - math.log(len(paths))


@torch.no_grad()
def evaluate(model, batches, num_batches, paths=None):
    """Exact token-mixture NLL with O(eval_tokens) CPU storage, no vocab cache."""
    if num_batches < 1 or num_batches > batches.num_batches:
        raise ValueError("Evaluation must traverse a nonempty held-out subset once")
    model.eval()
    device = next(model.parameters()).device
    members = paths if paths is not None else [None]
    if not members:
        raise ValueError("The ensemble has no snapshots")
    accumulated = [None] * num_batches
    individual = []
    for path in members:
        if path is not None:
            load_snapshot(model, path)
        loss_sum, count = 0.0, 0
        for i in range(num_batches):
            x, y = batches.batch(i, device)
            with autocast(device):
                logits = model(x)
            log_p = (
                -F.cross_entropy(logits.flatten(0, 1), y.flatten(), reduction="none")
                .cpu()
                .double()
            )
            loss_sum -= log_p.sum().item()
            count += log_p.numel()
            accumulated[i] = (
                log_p
                if accumulated[i] is None
                else torch.logaddexp(accumulated[i], log_p)
            )
        individual.append(loss_sum / count)
    loss = (
        -sum((item - math.log(len(members))).sum().item() for item in accumulated)
        / count
    )
    return {
        "loss": loss,
        "tokens": count,
        "members": len(members),
        "member_losses": individual,
    }

# =============================================================================
# Training
# =============================================================================

def main(argv=None):
    args = parser().parse_args(argv)
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", 0)
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    config = GPTConfig(
        sequence_len=args.sequence_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_kv_head=args.n_head,
        n_embd=args.n_embd,
    )
    val_data = TokenBatches(
        args.val_data, args.device_batch_size, args.sequence_len, shuffle=False
    )
    eval_batches = min(args.eval_batches or val_data.num_batches, val_data.num_batches)
    if args.eval_only:
        manifest = json.loads((args.output / "snapshots.json").read_text())
        model = GPT(GPTConfig(**manifest["model_config"])).to(device)
        model.init_weights()  # Install direction/gain parameters before loading samples.
        if model.config.sequence_len != args.sequence_len:
            raise ValueError("Evaluation sequence length must match the manifest")
        paths = [args.output / item["path"] for item in manifest["snapshots"]]
        result = evaluate(model, val_data, eval_batches, paths)
        write_json(result, args.output / "ensemble-eval.json")
        print(json.dumps(result), flush=True)
        return

    train_data = TokenBatches(
        args.train_data,
        args.device_batch_size,
        args.sequence_len,
        shuffle=True,
        seed=args.seed,
    )
    if train_data.identity["sha256"] == val_data.identity["sha256"]:
        raise ValueError("Training and held-out data must be separate")
    accumulation = args.total_batch_size // (args.device_batch_size * args.sequence_len)
    micro_limit = (
        args.num_epochs * train_data.num_batches
        if args.steps is None
        else args.steps * accumulation
    )
    total_steps = math.ceil(micro_limit / accumulation)
    schedule = snapshot_steps(total_steps, args.snapshots) if args.snapshots else []
    stop = min(args.stop_after or total_steps, total_steps)
    controller = LossBudgetController(
        target=args.loss_target,
        beta=args.loss_ema_beta,
        seed_std=args.noise_seed_std,
        max_std=args.noise_max_std,
        growth=args.noise_growth,
        interval=args.noise_interval,
        warmup=args.noise_warmup_steps,
        tolerance=args.loss_tolerance,
        enabled=not args.no_noise,
    )
    contract = {
        "model_config": asdict(config),
        "total_steps": total_steps,
        "micro_limit": micro_limit,
        "accumulation": accumulation,
        "batch_size": args.device_batch_size,
        "train_sha256": train_data.identity["sha256"],
        "val_sha256": val_data.identity["sha256"],
        "seed": args.seed,
        "controller_config": controller.state_dict(),
        "snapshot_steps": schedule,
        "optimizer": {
            k: getattr(args, k)
            for k in (
                "matrix_lr",
                "scale_lr",
                "scalar_lr",
                "embedding_lr",
                "unembedding_lr",
                "scale_weight_decay",
                "lr_warmup_steps",
                "final_lr_fraction",
            )
        },
    }
    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=True, mmap=True
        )
        if checkpoint["contract"] != contract:
            raise ValueError(
                "Resume configuration/data/horizon differ from the saved run"
            )
        if not (args.output / "snapshots.json").exists():
            raise ValueError(
                "Resume in the original output directory to retain ensemble members"
            )
        manifest = json.loads((args.output / "snapshots.json").read_text())
        if manifest["planned_steps"] != schedule or any(
            item["step"] > checkpoint["step"] for item in manifest["snapshots"]
        ):
            raise ValueError(
                "Snapshot manifest is ahead of or inconsistent with the resume checkpoint"
            )
        expected_saved = [s for s in schedule if s <= checkpoint["step"]]
        if [item["step"] for item in manifest["snapshots"]] != expected_saved or any(
            not (args.output / item["path"]).exists() for item in manifest["snapshots"]
        ):
            raise ValueError("Resume snapshot history is incomplete")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "baseline_commit": "52e7441f862c3295c0f5695933438dac78f7fc5b",
            "model_config": asdict(config),
            "planned_steps": schedule,
            "snapshots": [],
            "complete": False,
        }
        write_json(manifest, args.output / "snapshots.json")
    model = GPT(config).to(device)
    model.init_weights()
    muon, adam = model.setup_optimizer(
        matrix_lr=args.matrix_lr,
        scale_lr=args.scale_lr,
        embedding_lr=args.embedding_lr,
        scale_weight_decay=args.scale_weight_decay,
        compile=args.compile,
        unembedding_lr=args.unembedding_lr,
        scalar_lr=args.scalar_lr,
    )
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    step, micro_seen = 0, 0
    if args.resume:
        model.load_state_dict(checkpoint["model"], strict=True)
        muon.load_state_dict(checkpoint["muon"])
        adam.load_state_dict(checkpoint["adam"])
        controller = LossBudgetController(**checkpoint["controller"])
        noise_generator.set_state(checkpoint["noise_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if checkpoint["cuda_rng"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        step, micro_seen = checkpoint["step"], checkpoint["micro_seen"]
        del checkpoint
    forward_model = torch.compile(model) if args.compile else model
    environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        **git_info(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "source_sha256": {Path(__file__).name: file_sha256(__file__)},
    }
    if not args.resume:
        write_json(
            {
                "args": {
                    k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()
                },
                "environment": environment,
                "contract": contract,
                "data": {"train": train_data.identity, "val": val_data.identity},
            },
            args.output / "run.json",
        )
    print(
        json.dumps(
            {
                "environment": environment,
                "total_steps": total_steps,
                "snapshot_steps": schedule,
            }
        ),
        flush=True,
    )
    initial_eval = evaluate(model, val_data, eval_batches)
    print(json.dumps({"step": step, "validation": initial_eval}), flush=True)
    started = time.monotonic()
    with (args.output / "metrics.jsonl").open("a") as metrics:
        while step < stop:
            if device.type == "cuda":
                torch.cuda.synchronize()
            tick = time.monotonic()
            model.train()
            model.zero_grad(set_to_none=True)
            loss_sum = torch.zeros((), device=device)
            micro_count = min(accumulation, micro_limit - micro_seen)
            for _ in range(micro_count):
                x, y = train_data.batch(micro_seen, device)
                with autocast(device):
                    loss = forward_model(x, y)
                loss_sum += loss.detach()
                (loss / micro_count).backward()
                micro_seen += 1
            mean_loss = (loss_sum / micro_count).item()
            controller.observe(mean_loss)
            # Check scalar/Adam gradients and matrices once per step before any mutation.
            finite = torch.stack(
                [
                    p.grad.isfinite().all()
                    for p in model.parameters()
                    if p.grad is not None
                ]
            ).all()
            if not finite.item():
                raise FloatingPointError(
                    "Nonfinite gradients; optimizer update was not applied"
                )
            multiplier = lr_multiplier(
                step, total_steps, args.lr_warmup_steps, args.final_lr_fraction
            )
            for opt in (muon, adam):
                for group in opt.param_groups:
                    group["lr"] = group["initial_lr"] * multiplier
            for group in muon.param_groups:
                group["momentum"] = 0.85 + 0.1 * min(step / 300, 1)
            adam.step()
            project_embeddings_(model)
            muon.step(controller.noise_std, noise_generator)
            model.zero_grad(set_to_none=True)
            step += 1
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.monotonic() - tick
            record = {
                "step": step,
                "epoch": micro_seen / train_data.num_batches,
                "train_loss": mean_loss,
                "loss_ema": controller.ema,
                "noise_std": controller.noise_std,
                "applied_noise_std": controller.noise_std * math.sqrt(multiplier),
                "lr_multiplier": multiplier,
                "step_seconds": elapsed,
                "tokens": micro_count * args.device_batch_size * args.sequence_len,
            }
            if args.eval_every and step % args.eval_every == 0:
                record["validation"] = evaluate(model, val_data, eval_batches)
            if step in schedule:
                path = f"snapshot-{step:07d}.pt"
                atomic_save(
                    {
                        "model": cpu_state(model),
                        "step": step,
                        "model_config": asdict(config),
                    },
                    args.output / path,
                )
                manifest["snapshots"].append(
                    {"step": step, "path": path, "noise_std": controller.noise_std}
                )
                write_json(manifest, args.output / "snapshots.json")
            if args.save_every and step % args.save_every == 0:
                save_training(
                    args.output / "latest.pt",
                    model,
                    muon,
                    adam,
                    controller,
                    noise_generator,
                    step,
                    micro_seen,
                    contract,
                )
            metrics.write(json.dumps(record, allow_nan=False) + "\n")
            metrics.flush()
            print(json.dumps(record), flush=True)
    final_eval = evaluate(model, val_data, eval_batches)
    errors = sphere_errors(model)
    save_training(
        args.output / "latest.pt",
        model,
        muon,
        adam,
        controller,
        noise_generator,
        step,
        micro_seen,
        contract,
    )
    manifest["complete"] = (
        step == total_steps
        and [item["step"] for item in manifest["snapshots"]] == schedule
    )
    write_json(manifest, args.output / "snapshots.json")
    result = {
        "step": step,
        "planned_steps": total_steps,
        "completed": step == total_steps,
        "initial_validation": initial_eval,
        "final_validation": final_eval,
        "loss_ema": controller.ema,
        "noise_std": controller.noise_std,
        "sphere_error": errors,
        "snapshot_count": len(manifest["snapshots"]),
        "session_seconds": time.monotonic() - started,
        "peak_memory_gib": (
            torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
        ),
    }
    if manifest["snapshots"] and not args.skip_ensemble and step == total_steps:
        result["ensemble"] = evaluate(
            model,
            val_data,
            eval_batches,
            [args.output / item["path"] for item in manifest["snapshots"]],
        )
    write_json(result, args.output / "result.json")
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
