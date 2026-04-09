# ============================================================
# Dockerfile — Smart City Traffic Surveillance Analyst
# Matches passing repo pattern: pip install . via pyproject.toml
# ============================================================

FROM python:3.10.16-slim-bookworm

WORKDIR /app

# Copy package manifest first for Docker layer caching
COPY pyproject.toml .
COPY README.md .

# Install all dependencies via pyproject.toml (same as passing repo)
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir .

# Copy all source files
COPY . .

# HF Spaces requires port 7860
EXPOSE 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Health check targeting /healthz (matches openenv.yaml endpoints spec)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/healthz')" || exit 1

CMD ["python", "app.py"]
