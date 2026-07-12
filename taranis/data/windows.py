"""Fenêtrage et découpage chronologique.

Deux idées à retenir :

1. Chaque exemple d'entraînement est une **fenêtre** de longueur `Tw`, une
   photo des dernières mesures. À partir de cette fenêtre, on veut prédire si
   un orage arrive dans les `H` pas suivants (l'horizon).

2. Sur une série temporelle, on **ne mélange jamais** les indices au hasard.
   Un split chronologique protège contre la fuite d'information et reflète la
   contrainte réelle : au moment de la prédiction, on n'a accès qu'au passé.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CANAUX = ("pressure", "temp", "humidity", "wind")


@dataclass(frozen=True)
class WindowedDataset:
    """Dataset fenêtré, prêt à alimenter un modèle.

    Attributs
    ---------
    X : np.ndarray, forme (N, Tw, V)
        Les fenêtres. V = 4 canaux physiques dans l'ordre `CANAUX`.
    y : np.ndarray, forme (N,)
        Étiquettes binaires, 1 si au moins un onset d'orage tombe dans les
        `H` pas suivant la fenêtre.
    timestamps : np.ndarray, forme (N,)
        Horodatage de la **fin** de chaque fenêtre. Sert au split et au débogage.
    Tw : int
        Longueur de fenêtre (nombre de pas de temps).
    H : int
        Horizon de prédiction (nombre de pas de temps).
    """

    X: np.ndarray
    y: np.ndarray
    timestamps: np.ndarray
    Tw: int
    H: int

    def __len__(self) -> int:
        return len(self.y)


def make_windows(
    df: pd.DataFrame,
    Tw: int = 96,
    H: int = 48,
    stride: int = 1,
    label_col: str = "storm_onset",
    canaux: tuple[str, ...] = CANAUX,
) -> WindowedDataset:
    """Découpe une série temporelle en fenêtres, avec étiquette d'horizon.

    Pour chaque fin de fenêtre à l'indice `t`, la fenêtre couvre `[t-Tw+1, t]`
    et l'horizon couvre `[t+1, t+H]`. On étiquette `y=1` si `label_col` vaut
    True au moins une fois dans cet horizon, `y=0` sinon.

    Paramètres
    ----------
    df : pd.DataFrame
        Doit contenir les colonnes de `canaux` et `label_col`. Pas de temps
        régulier supposé.
    Tw : int
        Longueur de la fenêtre d'entrée.
    H : int
        Horizon de prédiction, en pas de temps.
    stride : int
        Décalage entre deux fenêtres successives. `1` maximise le nombre
        d'exemples, `Tw` supprime tout chevauchement.
    label_col : str
        Colonne booléenne utilisée pour construire `y`. Par défaut on prédit
        l'apparition d'un nouvel orage (`storm_onset`).
    canaux : tuple[str, ...]
        Colonnes d'entrée dans l'ordre voulu pour le tenseur X.

    Retour
    ------
    WindowedDataset
    """
    if Tw < 1 or H < 1:
        raise ValueError("Tw et H doivent être strictement positifs")
    if stride < 1:
        raise ValueError("stride doit être >= 1")
    n = len(df)
    if n < Tw + H:
        raise ValueError(
            f"série trop courte : {n} pas, il en faut au moins Tw+H={Tw + H}"
        )

    X_arr = df[list(canaux)].to_numpy(dtype=np.float32)
    label = df[label_col].to_numpy(dtype=bool)
    ts = df["timestamp"].to_numpy()

    # positions des fins de fenêtre valides, telles que l'horizon rentre encore
    ends = np.arange(Tw - 1, n - H, stride)
    N = len(ends)

    X = np.empty((N, Tw, len(canaux)), dtype=np.float32)
    y = np.empty(N, dtype=np.int64)
    end_ts = np.empty(N, dtype=ts.dtype)

    for i, t in enumerate(ends):
        X[i] = X_arr[t - Tw + 1 : t + 1]
        y[i] = int(label[t + 1 : t + 1 + H].any())
        end_ts[i] = ts[t]

    return WindowedDataset(X=X, y=y, timestamps=end_ts, Tw=Tw, H=H)


def chronological_split(
    ds: WindowedDataset,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> tuple[WindowedDataset, WindowedDataset, WindowedDataset]:
    """Découpe un `WindowedDataset` en train, val, test dans l'ordre du temps.

    Les ratios sont normalisés à 1. Le split se fait sur l'axe des fenêtres,
    dans l'ordre où elles ont été produites (chronologique). Aucun mélange.

    Retour
    ------
    (train, val, test)
    """
    if len(ds) == 0:
        raise ValueError("dataset vide")
    s = sum(ratios)
    r_train, r_val, r_test = (r / s for r in ratios)
    n = len(ds)
    n_train = int(n * r_train)
    n_val = int(n * r_val)
    # tout ce qui reste va dans test, pour ne perdre aucune fenêtre
    idx_train = slice(0, n_train)
    idx_val = slice(n_train, n_train + n_val)
    idx_test = slice(n_train + n_val, n)
    return (
        _slice(ds, idx_train),
        _slice(ds, idx_val),
        _slice(ds, idx_test),
    )


def _slice(ds: WindowedDataset, s: slice) -> WindowedDataset:
    return WindowedDataset(
        X=ds.X[s],
        y=ds.y[s],
        timestamps=ds.timestamps[s],
        Tw=ds.Tw,
        H=ds.H,
    )


def channel_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Moyenne et écart-type par canal, calculés sur l'ensemble des fenêtres.

    Utile pour normaliser train, val et test avec les stats du train seulement,
    ce qui évite toute fuite d'information.

    Note : sur de très grands datasets (millions d'échantillons), on force
    l'accumulation en float64 pour éviter la perte de précision d'un sum
    float32 (qui peut donner des résultats absurdes sur ~10^8 valeurs).
    """
    flat = X.reshape(-1, X.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalisation canal par canal, en float32."""
    return ((X - mean) / std).astype(np.float32)
