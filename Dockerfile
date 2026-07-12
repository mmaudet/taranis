FROM python:3.12-slim

# Image de serveur autonome Taranis.
# Contient : le paquet Python, la sonde entraînée, l'UI HTML, et uvicorn.
# Aucun état externe requis en fonctionnement. Les mesures SYNOP sont
# récupérées à la demande depuis l'API publique Opendatasoft.

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    TARANIS_PROBE=/app/runs/probe/combined

# uv pour installer les deps rapidement
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/uv

WORKDIR /app

# Dépendances Python en cache dédié (couche Docker stable)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --extra api

# Code applicatif
COPY taranis/ ./taranis/
COPY runs/probe/ ./runs/probe/

# Installe le paquet lui-même
RUN uv sync --frozen --extra api

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request as u; \
        u.urlopen('http://127.0.0.1:8000/health', timeout=3).read()" || exit 1

CMD ["uvicorn", "taranis.infer.api:app", "--host", "0.0.0.0", "--port", "8000"]
