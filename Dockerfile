FROM python:3.11-slim

# ── system dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libglib2.0-0 libsm6 libxrender1 libxext6 git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies (cached layer) ────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── install package ────────────────────────────────────────────────────────────
COPY . .
RUN pip install --no-cache-dir -e .

# ── expose ports ───────────────────────────────────────────────────────────────
EXPOSE 8000 8501

# ── default: start API ─────────────────────────────────────────────────────────
CMD ["uvicorn", "retail_assistant.api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "src"]
