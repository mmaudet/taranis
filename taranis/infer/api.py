"""API d'inférence FastAPI pour Taranis.

Trois endpoints :

- `POST /predict`  : accepte une fenêtre `(Tw, V)` en unités physiques,
                     retourne probabilité, niveau d'alerte, disclaimer.
- `GET  /health`   : sanity check + infos du modèle chargé.
- `GET  /stations/{id}/live` : récupère les 4 derniers jours SYNOP depuis
                     Opendatasoft pour la station demandée, calcule une
                     prédiction sur la fenêtre courante et retourne un
                     JSON prêt à afficher.

Lancement local :

    uv sync --extra api
    TARANIS_PROBE=runs/probe/ww_rich  \\
      uv run uvicorn taranis.infer.api:app --port 8000
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

try:
    import requests
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
except ImportError as e:  # pragma: no cover - installation optionnelle
    raise SystemExit(
        "fastapi/requests requis. Installer avec :\n"
        "  uv sync --extra api"
    ) from e

from taranis.data.meteofrance import CANAUX_MF_RICH, prepare_station, read_synop_csv
from taranis.infer.inference import Predictor

ROOT = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = Path(__file__).parent / "static"
PROBE_PATH = Path(os.environ.get("TARANIS_PROBE", ROOT / "runs" / "probe" / "ww_rich"))

app = FastAPI(
    title="Taranis Inference API",
    version="0.2",
    description="Nowcasting orage à partir de mesures capteur ponctuelles.",
)


# Sert les fichiers statiques du front sous /static (CSS/JS futurs)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    """Page HTML mobile-first, sélecteur de station et prédictions live."""
    index_html = STATIC_DIR / "index.html"
    if not index_html.exists():
        raise HTTPException(500, "index.html manquant dans le paquet installé")
    return FileResponse(index_html, media_type="text/html")

_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor(PROBE_PATH)
    return _predictor


# ---------------------------------------------------------------------------
# Schémas
# ---------------------------------------------------------------------------


class PredictIn(BaseModel):
    """Payload d'entrée pour /predict, fenêtre en unités physiques."""

    canaux: list[str] = Field(
        ..., description="Noms des canaux, doivent correspondre à ceux du modèle"
    )
    window: list[list[float]] = Field(
        ..., description="Tenseur (Tw, V) en unités physiques, ligne = pas de temps"
    )


class PredictOut(BaseModel):
    proba_orage: float
    alerte: str
    alerte_libelle: str
    alerte_couleur: str
    seuil_reference: float
    canaux: list[str]
    disclaimer: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    pred = get_predictor()
    return {
        "status": "ok",
        "canaux_attendus": pred.canaux,
        "Tw_pas": pred.config.Tw,
        "step_min_attendu": "3h (pour SYNOP)",
        "seuil_val_reference": pred.threshold,
    }


@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn):
    pred = get_predictor()
    if payload.canaux != pred.canaux:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Ordre de canaux différent. Attendu : {pred.canaux}, "
                f"reçu : {payload.canaux}"
            ),
        )
    X = np.asarray(payload.window, dtype=np.float32)
    try:
        out = pred.predict_from_raw(X)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return out.to_dict()


# ---------------------------------------------------------------------------
# Récupération SYNOP live
# ---------------------------------------------------------------------------


def _latest_synop_date(station_id: str):
    """Retourne la date SYNOP la plus récente disponible pour la station."""
    r = requests.get(
        "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
        "donnees-synop-essentielles-omm/records",
        params={
            "where": f"numer_sta='{station_id.zfill(5)}'",
            "select": "max(date) as fin",
            "limit": 1,
        },
        timeout=30,
    )
    r.raise_for_status()
    fin = r.json()["results"][0]["fin"]
    from datetime import datetime
    return datetime.fromisoformat(fin.replace("Z", "+00:00"))


def _fetch_recent_synop(station_id: str, days: int = 6, end=None) -> str:
    """Télécharge les mesures SYNOP jusqu'à `end` (ou dernière date dispo)."""
    from datetime import timedelta

    if end is None:
        end = _latest_synop_date(station_id)
    start = end - timedelta(days=days)
    r = requests.get(
        "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
        "donnees-synop-essentielles-omm/exports/csv",
        params={
            "where": (
                f"numer_sta='{station_id.zfill(5)}' "
                f"AND date >= '{start:%Y-%m-%d}' "
                f"AND date <= '{(end + timedelta(days=1)):%Y-%m-%d}'"
            ),
            "select": (
                "numer_sta,nom,altitude,date,pres,pmer,tc,u,ff,dd,rr1,rr3,raf10,ww"
            ),
            "delimiter": ";",
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.text


@app.get("/stations/{station_id}/live")
def station_live(station_id: str, with_window: bool = False):
    """Récupère les derniers jours SYNOP disponibles, calcule une prédiction et renvoie tout.

    Si `with_window=true`, la réponse inclut aussi la fenêtre par canal pour
    afficher les sparklines côté front.
    """
    from io import StringIO

    pred = get_predictor()
    Tw = pred.config.Tw

    try:
        csv_text = _fetch_recent_synop(station_id, days=6)
    except Exception as e:  # pragma: no cover - dépend du réseau
        raise HTTPException(status_code=502, detail=f"SYNOP indisponible : {e}") from e

    stations = read_synop_csv(StringIO(csv_text))
    if not stations:
        raise HTTPException(status_code=404, detail=f"aucune donnée pour {station_id}")

    (_sid, raw), = stations.items()
    d = prepare_station(raw, freq="3h")
    d = d.dropna(subset=list(CANAUX_MF_RICH)).reset_index(drop=True)
    if len(d) < Tw:
        raise HTTPException(
            status_code=422,
            detail=f"pas assez de mesures pour former une fenêtre ({len(d)}/{Tw})",
        )

    window = d.iloc[-Tw:]
    X = window[list(CANAUX_MF_RICH)].to_numpy(dtype=np.float32)
    prediction = pred.predict_from_raw(X)

    payload = {
        "station_id": station_id,
        "station_nom": raw["station_name"].iloc[0],
        "altitude_m": float(raw["altitude_m"].iloc[0]),
        "derniere_mesure": d["timestamp"].iloc[-1].isoformat(),
        "n_pas_utilises": Tw,
        "prediction": prediction.to_dict(),
    }
    if with_window:
        payload["window"] = {
            c: window[c].astype(float).tolist() for c in CANAUX_MF_RICH
        }
    return payload
