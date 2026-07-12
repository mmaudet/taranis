# Étape 5, TS-JEPA brique par brique

À l'étape 4, on a posé l'idée et l'architecture sur un schéma. À l'étape 5, on la traduit en code PyTorch, une brique à la fois. Chaque brique est petite, testée isolément, puis on assemble. Tout le code vit dans un seul fichier `taranis/models/tsjepa.py`, dans le même ordre que ce chapitre.

Convention : on note `B` la taille de batch, `T = 96` la longueur de fenêtre, `V = 4` le nombre de canaux physiques, `L = 8` la taille d'un patch, `N = T / L = 12` le nombre de patches par fenêtre, `D = 96` la dimension latente.

## 5a, PatchEmbed, découper et projeter

L'entrée est un tenseur `(B, T, V)`. Un transformer attend une séquence de tokens de dimension `D`. On casse donc la fenêtre en `N` patches, chacun contenant `L * V` valeurs, et on projette chaque patch linéairement vers `D`.

```python
class PatchEmbed(nn.Module):
    def __init__(self, patch_len: int, n_canaux: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.n_canaux = n_canaux
        self.proj = nn.Linear(patch_len * n_canaux, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, V = x.shape
        n_patches = T // self.patch_len
        x = x.view(B, n_patches, self.patch_len, V).reshape(B, n_patches, -1)
        return self.proj(x)
```

Deux tests suffisent : la forme, et le fait que deux patches réellement différents produisent deux embeddings différents (autrement dit, la projection ne mange pas l'information). Voir `tests/test_tsjepa.py::test_patch_embed_forme` et `test_patch_embed_deux_patches_differents_valeurs_differentes`.

## 5b, TransformerBlock, la brique commune

Rien d'original dans le bloc. On veut du pre-norm, une attention self, un MLP à ratio 2, deux résidus. C'est ce qu'utilisent la plupart des JEPA récents. La force de JEPA vient de la **procédure** d'entraînement, pas de la finesse du bloc.

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x
```

`TransformerEncoder` en empile `n_layers` et termine par une `LayerNorm`. Les deux tests unitaires : la forme est conservée, et le gradient traverse le bloc.

## 5c, sample_block_mask, choisir contexte et cible

Le masquage est le cœur de l'auto-supervision. On tire `n_blocks` positions de départ dans `[0, N - block_size]`, on rejette les configurations où deux blocs se chevauchent, et on sépare les patches en deux ensembles disjoints, contexte et cible.

```python
def sample_block_mask(n_patches, n_blocks=2, block_size=3, generator=None, max_tries=100):
    all_starts = torch.arange(n_patches - block_size + 1)
    for _ in range(max_tries):
        idx = torch.randperm(len(all_starts), generator=generator)[:n_blocks]
        starts, _ = torch.sort(all_starts[idx])
        gaps = starts[1:] - starts[:-1]
        if (gaps >= block_size).all():
            break
    else:
        raise RuntimeError(...)
    target_positions = torch.cat([torch.arange(s, s + block_size) for s in starts.tolist()])
    is_target = torch.zeros(n_patches, dtype=torch.bool)
    is_target[target_positions] = True
    context_idx = torch.nonzero(~is_target).squeeze(-1).long()
    return context_idx, target_positions.long()
```

Choix pédagogiques importants :

- **Un seul masque partagé sur tout le batch.** C'est plus simple à implémenter, à raisonner et à lire, sans gros impact tant que le batch est grand. Les implémentations plus avancées font varier le masque par échantillon.
- **Blocs non chevauchants.** Deux blocs identiques ou qui se recouvrent partiellement, ce serait de facto un seul bloc plus petit, et une variance de tâche moindre.
- **Fort taux de masquage.** Avec `n_blocks=2` et `block_size=3` sur `N=12`, la moitié des patches sont cachés. C'est ce qui rend la tâche non triviale.

Deux tests : le masque partitionne bien les positions (`test_masque_disjoint_et_complet`), et sur 20 seeds les blocs sont bien non chevauchants (`test_masque_blocs_non_chevauchants`).

## 5d, EMAWrapper, l'encodeur cible qui traîne derrière

C'est le point le plus subtil de JEPA. On veut deux encodeurs :

- l'un, dit **online**, qui reçoit les gradients à chaque étape,
- l'autre, dit **cible**, qui est une **moyenne mobile exponentielle** des paramètres du premier, et qui ne reçoit **jamais** de gradient.

À chaque pas d'entraînement :

$$\theta^{-} \leftarrow \tau \cdot \theta^{-} + (1 - \tau) \cdot \theta$$

Avec `τ = 0.996`, le cible met plusieurs milliers d'étapes à rattraper l'online. Cette lenteur est essentielle : sans elle, les deux encodeurs pourraient dégénérer ensemble vers la solution triviale « tout est zéro ».

```python
class EMAWrapper(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.encoder = copy.deepcopy(source)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, source, tau):
        for p_ema, p in zip(self.encoder.parameters(),
                            source.parameters(), strict=True):
            p_ema.data.mul_(tau).add_(p.data, alpha=1.0 - tau)

    def forward(self, x):
        with torch.no_grad():
            return self.encoder(x)
```

Deux tests : les paramètres du wrapper ne sont pas trainables (`test_ema_wrapper_ne_recoit_pas_gradient`), et l'EMA converge bien vers la source si on la met à jour assez de fois (`test_ema_wrapper_converge_vers_source`).

## 5e, TSJEPA, l'assemblage

On combine :

- un `PatchEmbed` pour les patches,
- une `nn.Embedding` pour les positions,
- un jeton appris `mask_token` que le prédicteur mettra sur les positions cibles,
- un `TransformerEncoder` `encoder` (online),
- un `EMAWrapper(encoder)` `target_encoder`,
- un `TransformerEncoder` `predictor`,
- une `LayerNorm` `target_norm` qui verrouille l'échelle de la cible.

Le forward suit exactement le schéma de l'étape 4 :

```python
def forward(self, x, context_idx, target_idx):
    z_target = self.encode_target(x, target_idx)     # EMA + LayerNorm + no_grad
    z_context = self.encode_context(x, context_idx)  # encodeur online
    pred_target = self.predict(z_context, context_idx, target_idx)
    return pred_target, z_target
```

Le prédicteur reçoit la concaténation `[z_context, mask_tokens + pos_target]`. En sortie, on garde uniquement les positions cibles, qui portent la prédiction. La perte est SmoothL1 entre `pred_target` et `z_target`.

Ce qu'on vérifie automatiquement :

- forme de la sortie (`test_tsjepa_forward_forme`),
- la perte **diminue** en 30 étapes sur des données aléatoires (`test_tsjepa_perte_diminue_sur_un_pas`). Ce test **suffit** pour se convaincre que la mécanique tourne, avant d'entamer un vrai entraînement.
- aucun paramètre du `target_encoder` ne reçoit de gradient (`test_target_encoder_ne_recoit_pas_gradient`). C'est le contrôle stop-gradient.
- l'utilitaire `embedding_stats` détecte bien un cas de collapse (`test_embedding_stats_detecte_collapse`) et donne un rang effectif proche de la dimension sur du bruit (`test_embedding_stats_sur_batch_aleatoire`). C'est la sonde qu'on branchera à l'étape 6.

## Ce qu'il faut retenir avant l'étape 6

1. Le modèle tient dans **un seul fichier**, ~250 lignes commentées.
2. Chaque brique est testée en isolation, et l'assemblage passe deux tests d'intégrité, forme et diminution de perte.
3. L'utilitaire `embedding_stats` est prêt à surveiller le collapse en continu.

À l'étape 6, on va :

- écrire une boucle d'entraînement qui itère sur `data/synthetic_windows.npz`,
- logger la perte, la stat d'écart-type et le rang effectif à chaque validation,
- écrire un test de non-régression sur le collapse.

Et si tout se passe bien, à l'étape 7, on branche une sonde linéaire et on compare M1 à la baseline M0.
