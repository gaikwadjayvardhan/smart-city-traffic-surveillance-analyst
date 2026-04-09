# ============================================================
# Dockerfile — Smart City Traffic Surveillance Analyst
# ============================================================
# Builds a clean FastAPI server exposed on port 7860 for
# HuggingFace Spaces automated pings (must return 200 OK on /).
#
# Build:  docker build -t traffic-env .
# Run:    docker run -p 7860:7860 \
#             -e OPENAI_API_KEY=sk-... \
#             -e MODEL_NAME=gpt-4o-mini \
#             traffic-env
# ============================================================

FROM python:3.11-slim

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies (installed before code copy for layer caching) ---
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# --- Application source ---
COPY . .

# --- Port & environment defaults ---
EXPOSE 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# --- Health check so HuggingFace Space marks the container as healthy ---
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# --- Entrypoint ---
CMD ["python", "app.py"]
