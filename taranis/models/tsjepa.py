"""TS-JEPA: JEPA adapted to time series.

This file contains all the TS-JEPA building blocks, in doc order:

1. PatchEmbed          : split a window into patches, project to D.
2. TransformerBlock    : reusable attention + MLP block.
3. TransformerEncoder  : stack of blocks with a final LayerNorm.
4. sample_block_mask   : non-overlapping block masking.
5. EMAWrapper          : target encoder, exponential moving average of the
                         online encoder parameters.
6. TSJEPA              : full assembly, ready to train.

Teaching notes:

- The online encoder (`fθ`) receives gradients. It is the one we keep for
  the downstream probe of step 7.
- The target encoder (`fθ-`) is an EMA copy of `fθ`, without gradient. It
  provides the "ground truth" the predictor tries to match.
- The loss is computed in latent space, on the target patches only.

See `docs/04-jepa-idee.md` for intuition and `docs/05-tsjepa-briques.md`
for the block-by-block code walkthrough.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

# --------------------------------------------------------------------------- #
# 1. PatchEmbed
# --------------------------------------------------------------------------- #


class PatchEmbed(nn.Module):
    """Split a `(B, T, V)` window into patches and project to D.

    - `T` must be a multiple of `patch_len`.
    - Each patch holds `patch_len` time steps of `V` channels, i.e.
      `patch_len * V` values, linearly projected to D.

    Output: `(B, n_patches, D)` with `n_patches = T / patch_len`.
    """

    def __init__(self, patch_len: int, n_canaux: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.n_canaux = n_canaux
        self.d_model = d_model
        self.proj = nn.Linear(patch_len * n_canaux, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, V = x.shape
        if T % self.patch_len != 0:
            raise ValueError(f"T={T} doit être multiple de patch_len={self.patch_len}")
        if V != self.n_canaux:
            raise ValueError(f"attendu {self.n_canaux} canaux, obtenu {V}")
        n_patches = T // self.patch_len
        # (B, T, V) -> (B, n_patches, patch_len, V) -> (B, n_patches, patch_len*V)
        x = x.view(B, n_patches, self.patch_len, V).reshape(B, n_patches, -1)
        return self.proj(x)


# --------------------------------------------------------------------------- #
# 2. TransformerBlock et 3. TransformerEncoder
# --------------------------------------------------------------------------- #


class TransformerBlock(nn.Module):
    """Standard transformer block: pre-norm attention + pre-norm MLP.

    Two residuals, a single self-attention, an MLP at ratio 2 by default.
    Nothing original, on purpose: the JEPA structure carries the learning,
    not the block architecture.
    """

    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of `n_layers` blocks followed by a final LayerNorm."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, mlp_ratio) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


# --------------------------------------------------------------------------- #
# 4. Block masking
# --------------------------------------------------------------------------- #


def sample_block_mask(
    n_patches: int,
    n_blocks: int = 2,
    block_size: int = 3,
    generator: torch.Generator | None = None,
    max_tries: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a non-overlapping block mask, shared across the batch.

    Draw `n_blocks` start positions in `[0, n_patches - block_size]`,
    rejecting any configuration where two blocks overlap. Patches covered
    by the blocks form the **target** (masked), the rest form the
    **context**.

    Returns
    -------
    (context_idx, target_idx): two `LongTensor` of complementary sizes such
    that `sorted(cat) == arange(n_patches)`.
    """
    if n_blocks * block_size > n_patches:
        raise ValueError(
            f"impossible de placer {n_blocks} blocs de taille {block_size} "
            f"dans {n_patches} patches"
        )

    all_starts = torch.arange(n_patches - block_size + 1)
    for _ in range(max_tries):
        idx = torch.randperm(len(all_starts), generator=generator)[:n_blocks]
        starts, _ = torch.sort(all_starts[idx])
        gaps = starts[1:] - starts[:-1]
        if (gaps >= block_size).all():
            break
    else:
        raise RuntimeError(
            f"impossible d'échantillonner {n_blocks} blocs disjoints après "
            f"{max_tries} tentatives"
        )

    target_positions = torch.cat(
        [torch.arange(s, s + block_size) for s in starts.tolist()]
    )
    is_target = torch.zeros(n_patches, dtype=torch.bool)
    is_target[target_positions] = True
    target_idx = target_positions.long()
    context_idx = torch.nonzero(~is_target, as_tuple=False).squeeze(-1).long()
    return context_idx, target_idx


# --------------------------------------------------------------------------- #
# 5. Encodeur cible EMA
# --------------------------------------------------------------------------- #


class EMAWrapper(nn.Module):
    """EMA copy of a module, without gradient.

    - In the constructor: deep copy the source, freeze all parameters.
    - At each `update(source, tau)`: each parameter is updated as
      `p = tau * p + (1 - tau) * p_source`.
    - `forward`: run the internal copy under `torch.no_grad`.

    No gradient should ever flow through this object.
    """

    def __init__(self, source: nn.Module):
        super().__init__()
        self.encoder = copy.deepcopy(source)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, source: nn.Module, tau: float) -> None:
        for p_ema, p in zip(self.encoder.parameters(), source.parameters(), strict=True):
            p_ema.data.mul_(tau).add_(p.data, alpha=1.0 - tau)
        for b_ema, b in zip(self.encoder.buffers(), source.buffers(), strict=True):
            b_ema.data.copy_(b.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.encoder(x)


# --------------------------------------------------------------------------- #
# 6. TS-JEPA, assemblage
# --------------------------------------------------------------------------- #


@dataclass
class TSJEPAConfig:
    Tw: int = 96
    n_canaux: int = 4
    patch_len: int = 8
    d_model: int = 96
    n_layers_encoder: int = 3
    n_layers_predictor: int = 2
    n_heads: int = 4
    mlp_ratio: float = 2.0
    tau_ema: float = 0.996

    @property
    def n_patches(self) -> int:
        return self.Tw // self.patch_len


class TSJEPA(nn.Module):
    """Full TS-JEPA model.

    Usage:

        model = TSJEPA(config)
        # each batch:
        ctx_idx, tgt_idx = sample_block_mask(config.n_patches, ...)
        pred_tgt, z_tgt = model(x, ctx_idx, tgt_idx)
        loss = F.smooth_l1_loss(pred_tgt, z_tgt)
        loss.backward()
        optimizer.step()
        model.update_target()   # EMA update after the step

    After training, the `encoder` is the only object needed for the
    downstream probe (step 7). Retrieve it via `model.freeze_encoder()`.
    """

    def __init__(self, config: TSJEPAConfig):
        super().__init__()
        self.config = config

        self.patch_embed = PatchEmbed(
            patch_len=config.patch_len,
            n_canaux=config.n_canaux,
            d_model=config.d_model,
        )
        self.pos_embed = nn.Embedding(config.n_patches, config.d_model)
        # learned token placed at target positions on predictor input
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.encoder = TransformerEncoder(
            d_model=config.d_model,
            n_layers=config.n_layers_encoder,
            n_heads=config.n_heads,
            mlp_ratio=config.mlp_ratio,
        )
        self.target_encoder = EMAWrapper(self.encoder)
        self.predictor = TransformerEncoder(
            d_model=config.d_model,
            n_layers=config.n_layers_predictor,
            n_heads=config.n_heads,
            mlp_ratio=config.mlp_ratio,
        )
        # LayerNorm on target: locks the scale, counter-measure against collapse
        self.target_norm = nn.LayerNorm(config.d_model)

    def _add_position(self, patches: torch.Tensor) -> torch.Tensor:
        """Add position embeddings to the patches."""
        B, N, _ = patches.shape
        positions = torch.arange(N, device=patches.device)
        return patches + self.pos_embed(positions).unsqueeze(0)

    def encode_context(self, x: torch.Tensor, context_idx: torch.Tensor) -> torch.Tensor:
        """Online encoding on the context patches only.

        Returns: `(B, n_ctx, D)`.
        """
        p = self._add_position(self.patch_embed(x))
        return self.encoder(p[:, context_idx, :])

    @torch.no_grad()
    def encode_target(
        self, x: torch.Tensor, target_idx: torch.Tensor
    ) -> torch.Tensor:
        """Target embeddings via the EMA encoder on the full sequence,
        then LayerNorm and selection of the target patches.

        Returns: `(B, n_tgt, D)`.
        """
        p = self._add_position(self.patch_embed(x))
        z_full = self.target_encoder(p)
        return self.target_norm(z_full[:, target_idx, :])

    def predict(
        self,
        z_context: torch.Tensor,
        context_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the target patch embeddings from the context.

        Concatenate `[z_context, mask_tokens + pos_target]` and feed it to
        the predictor. Return only the target-side output.
        """
        B = z_context.size(0)
        n_ctx = context_idx.shape[0]
        n_tgt = target_idx.shape[0]

        pos_tgt = self.pos_embed(target_idx.to(z_context.device))  # (n_tgt, D)
        tgt_tokens = self.mask_token.expand(B, n_tgt, -1) + pos_tgt.unsqueeze(0)

        seq = torch.cat([z_context, tgt_tokens], dim=1)  # (B, n_ctx + n_tgt, D)
        out = self.predictor(seq)
        return out[:, n_ctx:, :]  # (B, n_tgt, D)

    def forward(
        self,
        x: torch.Tensor,
        context_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One full step.

        Returns: `(pred_target, z_target)`, both shaped `(B, n_tgt, D)`.
        Typical loss: `F.smooth_l1_loss(pred_target, z_target)`.
        """
        z_target = self.encode_target(x, target_idx)  # stop-gradient inside
        z_context = self.encode_context(x, context_idx)
        pred_target = self.predict(z_context, context_idx, target_idx)
        return pred_target, z_target

    @torch.no_grad()
    def update_target(self, tau: float | None = None) -> None:
        """EMA update of the target encoder from the online encoder."""
        self.target_encoder.update(self.encoder, tau or self.config.tau_ema)

    def freeze_encoder(self) -> nn.Module:
        """Return the frozen online encoder, ready for the downstream probe."""
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        return self.encoder


# --------------------------------------------------------------------------- #
# Collapse-monitoring utilities (used at step 6)
# --------------------------------------------------------------------------- #


def embedding_stats(z: torch.Tensor) -> dict[str, float]:
    """Batch-level embedding statistics used to detect collapse.

    - `std_moy`  : mean per-dimension standard deviation across all tokens.
    - `eff_rank` : effective rank of the covariance, `exp(H(spectrum))`.
                   A value close to `D` indicates good dispersion; a value
                   close to 1 signals collapse.

    Use during training on `z_context` or `z_target`.
    """
    z_flat = z.reshape(-1, z.size(-1)).detach().float()
    std_moy = z_flat.std(dim=0).mean().item()

    z_c = z_flat - z_flat.mean(dim=0, keepdim=True)
    n = max(1, z_c.size(0) - 1)
    cov = (z_c.T @ z_c) / n
    eig = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    total = eig.sum()
    if total < 1e-10:
        # variance fully collapsed, effective rank ~ 1
        return {"std_moy": std_moy, "eff_rank": 1.0}
    p = (eig / total).clamp(min=1e-12)
    eff_rank = torch.exp(-(p * p.log()).sum()).item()
    return {"std_moy": std_moy, "eff_rank": eff_rank}


def jepa_loss(pred_target: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """SmoothL1 loss in latent space, on the target tokens."""
    return F.smooth_l1_loss(pred_target, z_target)
