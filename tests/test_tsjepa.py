"""Tests unitaires par brique + assemblage TS-JEPA."""

from __future__ import annotations

import torch

from taranis.models import (
    TSJEPA,
    EMAWrapper,
    PatchEmbed,
    TransformerBlock,
    TransformerEncoder,
    TSJEPAConfig,
    embedding_stats,
    jepa_loss,
    sample_block_mask,
)

# --- 5a. PatchEmbed --- #


def test_patch_embed_forme():
    B, T, V, L, D = 4, 24, 4, 4, 16
    layer = PatchEmbed(patch_len=L, n_canaux=V, d_model=D)
    x = torch.randn(B, T, V)
    out = layer(x)
    assert out.shape == (B, T // L, D)


def test_patch_embed_deux_patches_differents_valeurs_differentes():
    """Preuve que deux patches distincts donnent des embeddings distincts,
    autrement dit PatchEmbed n'écrase pas l'information."""
    torch.manual_seed(0)
    layer = PatchEmbed(patch_len=4, n_canaux=2, d_model=8)
    x = torch.zeros(1, 8, 2)
    x[0, :4] = 1.0
    x[0, 4:] = -1.0
    out = layer(x)
    assert out.shape == (1, 2, 8)
    assert not torch.allclose(out[0, 0], out[0, 1])


# --- 5b. TransformerBlock et TransformerEncoder --- #


def test_transformer_block_conserve_forme_et_transmet_gradient():
    B, N, D = 2, 6, 16
    block = TransformerBlock(d_model=D, n_heads=4)
    x = torch.randn(B, N, D, requires_grad=True)
    out = block(x)
    assert out.shape == x.shape
    out.sum().backward()
    assert x.grad is not None
    assert (x.grad.abs() > 0).any()


def test_transformer_encoder_multiples_couches():
    enc = TransformerEncoder(d_model=16, n_layers=3, n_heads=4)
    x = torch.randn(2, 6, 16)
    out = enc(x)
    assert out.shape == x.shape


# --- 5c. Masquage par blocs --- #


def test_masque_disjoint_et_complet():
    n_patches = 12
    ctx, tgt = sample_block_mask(n_patches, n_blocks=2, block_size=3)
    all_idx = torch.cat([ctx, tgt]).sort().values
    assert torch.equal(all_idx, torch.arange(n_patches))
    assert len(tgt) == 2 * 3
    assert len(ctx) == n_patches - 2 * 3


def test_masque_blocs_non_chevauchants():
    """Sur plusieurs seeds, on vérifie deux propriétés :
    - la cible contient exactement `n_blocks * block_size` positions uniques,
    - les blocs ne se chevauchent pas (au plus un « grand » saut entre blocs).
    """
    n_blocks, block_size = 2, 3
    for seed in range(20):
        g = torch.Generator().manual_seed(seed)
        _, tgt = sample_block_mask(n_patches=12, n_blocks=n_blocks, block_size=block_size, generator=g)
        tgt_sorted = tgt.sort().values.tolist()
        # bonne taille et pas de doublon
        assert len(tgt_sorted) == n_blocks * block_size
        assert len(set(tgt_sorted)) == n_blocks * block_size
        # les blocs ne se chevauchent pas : par construction, deux positions
        # consécutives dans la même série ne peuvent pas être plus rapprochées
        # que dans un bloc unique.
        diffs = [tgt_sorted[i + 1] - tgt_sorted[i] for i in range(len(tgt_sorted) - 1)]
        assert min(diffs) >= 1
        # au plus n_blocks - 1 sauts inter-blocs
        big_jumps = [d for d in diffs if d > 1]
        assert len(big_jumps) <= n_blocks - 1


# --- 5d. EMAWrapper --- #


def test_ema_wrapper_ne_recoit_pas_gradient():
    torch.manual_seed(0)
    enc = TransformerEncoder(d_model=16, n_layers=2, n_heads=4)
    ema = EMAWrapper(enc)
    x = torch.randn(2, 6, 16)
    out = ema(x)
    assert out.grad_fn is None
    for p in ema.parameters():
        assert not p.requires_grad


def test_ema_wrapper_converge_vers_source():
    torch.manual_seed(1)
    source = TransformerEncoder(d_model=16, n_layers=2, n_heads=4)
    ema = EMAWrapper(source)
    # perturber source
    with torch.no_grad():
        for p in source.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    # avant update : ema != source
    diff_before = sum(
        (a - b).abs().mean().item()
        for a, b in zip(ema.encoder.parameters(), source.parameters(), strict=True)
    )
    for _ in range(200):
        ema.update(source, tau=0.9)
    diff_after = sum(
        (a - b).abs().mean().item()
        for a, b in zip(ema.encoder.parameters(), source.parameters(), strict=True)
    )
    assert diff_after < diff_before * 0.05


# --- 5e. TSJEPA complet --- #


def _small_config():
    return TSJEPAConfig(
        Tw=24,
        n_canaux=4,
        patch_len=4,
        d_model=16,
        n_layers_encoder=2,
        n_layers_predictor=2,
        n_heads=4,
    )


def test_tsjepa_forward_forme():
    cfg = _small_config()
    model = TSJEPA(cfg)
    x = torch.randn(3, cfg.Tw, cfg.n_canaux)
    ctx, tgt = sample_block_mask(cfg.n_patches, n_blocks=2, block_size=2)
    pred, z_tgt = model(x, ctx, tgt)
    assert pred.shape == (3, len(tgt), cfg.d_model)
    assert z_tgt.shape == (3, len(tgt), cfg.d_model)


def test_tsjepa_perte_diminue_sur_un_pas():
    torch.manual_seed(42)
    cfg = _small_config()
    model = TSJEPA(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    x = torch.randn(8, cfg.Tw, cfg.n_canaux)
    ctx, tgt = sample_block_mask(cfg.n_patches, n_blocks=2, block_size=2)

    losses = []
    for _ in range(30):
        opt.zero_grad()
        pred, z_tgt = model(x, ctx, tgt)
        loss = jepa_loss(pred, z_tgt)
        loss.backward()
        opt.step()
        model.update_target(tau=0.9)
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.8, f"la perte doit descendre : {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_target_encoder_ne_recoit_pas_gradient():
    cfg = _small_config()
    model = TSJEPA(cfg)
    x = torch.randn(2, cfg.Tw, cfg.n_canaux)
    ctx, tgt = sample_block_mask(cfg.n_patches, n_blocks=2, block_size=2)
    pred, z_tgt = model(x, ctx, tgt)
    loss = jepa_loss(pred, z_tgt)
    loss.backward()
    # aucun paramètre de target_encoder ne doit avoir de gradient
    for p in model.target_encoder.parameters():
        assert p.grad is None


def test_embedding_stats_sur_batch_aleatoire():
    """Sur un grand batch de bruit gaussien, std ~ 1 et le rang effectif
    approche la dimension D."""
    torch.manual_seed(0)
    D = 32
    z = torch.randn(64, 32, D)  # 2048 échantillons, largement > D
    s = embedding_stats(z)
    assert 0.9 < s["std_moy"] < 1.1
    assert s["eff_rank"] > 0.8 * D  # proche de D


def test_embedding_stats_detecte_collapse():
    """Si tous les embeddings sont constants, std=0 et rang effectif = 1."""
    z = torch.ones(4, 10, 32) * 0.5
    s = embedding_stats(z)
    assert s["std_moy"] < 1e-6
    assert s["eff_rank"] < 1.5
