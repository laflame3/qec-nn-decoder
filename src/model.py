"""Neural decoders for E1 (PLAN v1.2 §1.5): Transformer and MLP, plus training.

Both models map one shot's detector events (n_det bits) to a logit for
"the logical observable flipped". Training samples are drawn fresh from Stim
for every batch, so a model never sees the same shot twice; the training
budget is the total number of shots seen (n_train), the same for all models.
"""

import math
import time

import numpy as np
import torch
from torch import nn


def detector_layout(circuit):
    """Integer stabilizer index and round index for every detector.

    Read from Stim's detector coordinates (x, y, t): each distinct (x, y) is
    one stabilizer, t is the round.
    """
    coords = circuit.get_detector_coordinates()
    n = circuit.num_detectors
    xy = [tuple(coords[i][:2]) for i in range(n)]
    t = [int(round(coords[i][2])) for i in range(n)]
    stab_ids = {v: k for k, v in enumerate(sorted(set(xy)))}
    stab = np.array([stab_ids[v] for v in xy])
    rnd = np.array(t) - min(t)
    return stab, rnd


class TransformerDecoder(nn.Module):
    """One token per detector: value + stabilizer-position + round embeddings.

    A learned CLS token is prepended; its final state goes through a linear
    head to give the logical-flip logit.
    """

    def __init__(self, stab, rnd, d_model=64, n_heads=4, n_layers=3, d_ff=128):
        super().__init__()
        self.register_buffer("stab", torch.as_tensor(stab, dtype=torch.long))
        self.register_buffer("rnd", torch.as_tensor(rnd, dtype=torch.long))
        self.value_emb = nn.Embedding(2, d_model)
        self.stab_emb = nn.Embedding(int(stab.max()) + 1, d_model)
        self.round_emb = nn.Embedding(int(rnd.max()) + 1, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):  # x: (batch, n_det) in {0, 1}
        tok = self.value_emb(x.long()) + self.stab_emb(self.stab) + self.round_emb(self.rnd)
        tok = torch.cat([self.cls.expand(x.shape[0], -1, -1), tok], dim=1)
        h = self.encoder(tok)
        return self.head(self.norm(h[:, 0])).squeeze(-1)


class MLPDecoder(nn.Module):
    """Flattened detector vector -> fully connected layers -> logit.

    Knows nothing about which detector sits where; the comparison with the
    Transformer separates "architecture" from "using a neural network".
    """

    def __init__(self, n_det, hidden=256, n_layers=3):
        super().__init__()
        layers, width = [], n_det
        for _ in range(n_layers):
            layers += [nn.Linear(width, hidden), nn.GELU()]
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float()).squeeze(-1)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def default_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@torch.no_grad()
def predict(model, detectors, device, batch=65536):
    model.eval()
    out = []
    for i in range(0, len(detectors), batch):
        x = torch.as_tensor(detectors[i : i + batch], device=device)
        out.append((model(x) > 0).cpu().numpy())
    return np.concatenate(out)


def train(
    model,
    circuit,
    val_det,
    val_obs,
    *,
    n_train=10_000_000,
    batch=1024,
    lr=1e-3,
    weight_decay=1e-4,
    chunk=500_000,
    sample_seed=0,
    torch_seed=0,
    device=None,
    log=print,
):
    """Train on fresh Stim shots; keep the weights with the best validation LER.

    Shots are drawn in chunks of `chunk` from a sampler seeded by sample_seed;
    validation LER is checked after every chunk. Learning rate: linear warmup
    over the first 2% of steps, then cosine decay to zero.
    """
    device = device or default_device()
    torch.manual_seed(torch_seed)
    model.to(device)
    sampler = circuit.compile_detector_sampler(seed=sample_seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = math.ceil(n_train / batch)
    warmup = max(1, int(0.02 * total_steps))

    def lr_at(step):
        if step < warmup:
            return lr * (step + 1) / warmup
        frac = (step - warmup) / max(1, total_steps - warmup)
        return lr * 0.5 * (1 + math.cos(math.pi * frac))

    loss_fn = nn.BCEWithLogitsLoss()
    best = (float("inf"), None)
    history = []
    step, seen, t0 = 0, 0, time.time()
    while seen < n_train:
        n = min(chunk, n_train - seen)
        det, obs = sampler.sample(n, separate_observables=True)
        perm = torch.randperm(n)
        det_t = torch.as_tensor(det, device=device)[perm.to(device)]
        obs_t = torch.as_tensor(obs[:, 0], device=device, dtype=torch.float32)[perm.to(device)]
        model.train()
        run_loss = 0.0
        for i in range(0, n, batch):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            logit = model(det_t[i : i + batch])
            loss = loss_fn(logit, obs_t[i : i + batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(logit)
            step += 1
        seen += n
        val_ler = float((predict(model, val_det, device) != val_obs).mean())
        history.append(
            {"seen": seen, "train_loss": run_loss / n, "val_ler": val_ler, "time_s": time.time() - t0}
        )
        log(f"  seen={seen:>9,d} loss={run_loss / n:.4f} val_LER={val_ler:.5f} ({time.time() - t0:.0f}s)")
        if val_ler < best[0]:
            best = (val_ler, {k: v.detach().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    return history
