"""TS-JEPA, adaptation de JEPA aux séries temporelles.

Ce fichier contient toutes les briques de TS-JEPA, dans l'ordre de la doc :

1. PatchEmbed              : découpe une fenêtre en patches et projette en D.
2. TransformerBlock        : bloc attention + MLP, réutilisable.
3. TransformerEncoder      : empilement de blocs, avec LayerNorm finale.
4. sample_block_mask       : masquage par blocs non chevauchants.
5. EMAWrapper              : encodeur cible, moyenne mobile exponentielle
                              des paramètres de l'encodeur online.
6. TSJEPA                  : assemblage complet, prête à entraîner.

Repères pédagogiques :

- L'encodeur online (`fθ`) reçoit les gradients. C'est celui qu'on garde
  pour la sonde aval de l'étape 7.
- L'encodeur cible (`fθ⁻`) est une copie EMA de `fθ`, sans gradient. Il fournit
  la « vérité » vers laquelle le prédicteur doit tendre.
- La perte se calcule dans l'espace latent, sur les patches cibles uniquement.

Voir `docs/04-jepa-idee.md` pour l'intuition et `docs/05-tsjepa-briques.md`
pour le fil de code brique par brique.
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
    """Découpe une fenêtre `(B, T, V)` en patches et projette en D.

    - `T` doit être un multiple de `patch_len`.
    - Un patch contient `patch_len` pas de temps de `V` canaux, soit
      `patch_len * V` valeurs, projetées linéairement en D.

    Sortie : `(B, n_patches, D)` avec `n_patches = T / patch_len`.
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
    """Bloc transformer standard : pre-norm attention + pre-norm MLP.

    Deux résidus, une seule attention self, un MLP à ratio 2 par défaut. Rien
    d'original, c'est délibéré : la structure JEPA porte l'apprentissage, pas
    l'architecture du bloc.
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
    """Empilement de `n_layers` blocs, suivi d'une LayerNorm finale."""

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
# 4. Masquage par blocs
# --------------------------------------------------------------------------- #


def sample_block_mask(
    n_patches: int,
    n_blocks: int = 2,
    block_size: int = 3,
    generator: torch.Generator | None = None,
    max_tries: int = 100,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tire un masque par blocs non chevauchants, partagé sur le batch.

    On tire `n_blocks` positions de départ dans `[0, n_patches - block_size]`,
    en rejetant les configurations où deux blocs se chevauchent. Les patches
    couverts par les blocs forment la **cible** (masquée), les autres forment
    le **contexte**.

    Retour
    ------
    (context_idx, target_idx) : deux tenseurs `LongTensor` de tailles complémentaires,
    tels que `sorted(cat) == arange(n_patches)`.
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
    """Copie EMA d'un module, sans gradient.

    - Au constructeur : deep copy de la source, gel de tous les paramètres.
    - À chaque `update(source, tau)` : chaque paramètre est mis à jour
      `p ← tau * p + (1 - tau) * p_source`.
    - `forward` : appelle la copie interne sous `torch.no_grad`.

    On ne veut jamais que du gradient traverse cet objet.
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
    """Modèle TS-JEPA complet.

    Interface d'usage :

        model = TSJEPA(config)
        # à chaque batch :
        ctx_idx, tgt_idx = sample_block_mask(config.n_patches, ...)
        pred_tgt, z_tgt = model(x, ctx_idx, tgt_idx)
        loss = F.smooth_l1_loss(pred_tgt, z_tgt)
        loss.backward()
        optimizer.step()
        model.update_target()   # mise à jour EMA après l'étape

    Après entraînement, l'encodeur `encoder` est le seul objet nécessaire à
    la sonde aval de l'étape 7. On le récupère via `model.freeze_encoder()`.
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
        # jeton appris qu'on met sur les positions cibles en entrée du prédicteur
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
        # LayerNorm sur la cible : verrouille l'échelle, contre-mesure au collapse
        self.target_norm = nn.LayerNorm(config.d_model)

    def _add_position(self, patches: torch.Tensor) -> torch.Tensor:
        """Ajoute les embeddings de position aux patches."""
        B, N, _ = patches.shape
        positions = torch.arange(N, device=patches.device)
        return patches + self.pos_embed(positions).unsqueeze(0)

    def encode_context(self, x: torch.Tensor, context_idx: torch.Tensor) -> torch.Tensor:
        """Encodage online sur les patches de contexte uniquement.

        Retour : `(B, n_ctx, D)`.
        """
        p = self._add_position(self.patch_embed(x))
        return self.encoder(p[:, context_idx, :])

    @torch.no_grad()
    def encode_target(
        self, x: torch.Tensor, target_idx: torch.Tensor
    ) -> torch.Tensor:
        """Embeddings cibles via l'encodeur EMA sur la séquence complète,
        puis LayerNorm et sélection des patches cibles.

        Retour : `(B, n_tgt, D)`.
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
        """Prédit les embeddings des patches cibles à partir du contexte.

        On concatène `[z_context, mask_tokens + pos_target]` et on passe le
        tout dans le prédicteur. On extrait la portion cible en sortie.
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
        """Un pas complet.

        Retour : `(pred_target, z_target)`, tous deux `(B, n_tgt, D)`.
        La perte typique est `F.smooth_l1_loss(pred_target, z_target)`.
        """
        z_target = self.encode_target(x, target_idx)  # stop-gradient dedans
        z_context = self.encode_context(x, context_idx)
        pred_target = self.predict(z_context, context_idx, target_idx)
        return pred_target, z_target

    @torch.no_grad()
    def update_target(self, tau: float | None = None) -> None:
        """Mise à jour EMA de l'encodeur cible depuis l'encodeur online."""
        self.target_encoder.update(self.encoder, tau or self.config.tau_ema)

    def freeze_encoder(self) -> nn.Module:
        """Renvoie l'encodeur online gelé, prêt pour la sonde aval."""
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        return self.encoder


# --------------------------------------------------------------------------- #
# Utilitaires de surveillance du collapse (utilisés à l'étape 6)
# --------------------------------------------------------------------------- #


def embedding_stats(z: torch.Tensor) -> dict[str, float]:
    """Statistiques d'un batch d'embeddings, pour détecter le collapse.

    - `std_moy`   : écart-type moyen par dimension, sur tous les tokens.
    - `eff_rank`  : rang effectif de la covariance, `exp(H(spectrum))`.
                     Une valeur proche de `D` indique une bonne dispersion,
                     une valeur proche de 1 indique un collapse.

    À utiliser en cours d'entraînement, sur `z_context` ou `z_target`.
    """
    z_flat = z.reshape(-1, z.size(-1)).detach().float()
    std_moy = z_flat.std(dim=0).mean().item()

    z_c = z_flat - z_flat.mean(dim=0, keepdim=True)
    n = max(1, z_c.size(0) - 1)
    cov = (z_c.T @ z_c) / n
    eig = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    total = eig.sum()
    if total < 1e-10:
        # variance totalement effondrée, rang effectif ≈ 1
        return {"std_moy": std_moy, "eff_rank": 1.0}
    p = (eig / total).clamp(min=1e-12)
    eff_rank = torch.exp(-(p * p.log()).sum()).item()
    return {"std_moy": std_moy, "eff_rank": eff_rank}


def jepa_loss(pred_target: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    """Perte SmoothL1 en espace latent, sur les tokens cibles."""
    return F.smooth_l1_loss(pred_target, z_target)
