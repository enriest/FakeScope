FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# System deps for building some wheels and lxml parser
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential git \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Copy app source
COPY src ./src

# DO NOT copy model - it will be downloaded from Hugging Face at runtime
# to keep Docker image small (< 1GB instead of 8GB+)
# Set FAKESCOPE_MODEL_DIR environment variable to your HF repo (e.g., "enriest/distilbert-fakenews")

# Optional: Create data dir for sqlite
RUN mkdir -p /app/data

EXPOSE 8080
EXPOSE 8001

# Run FastAPI (port 8001) and Streamlit (port 8080)
CMD ["/bin/sh", "-lc", "uvicorn src.api:app --host 0.0.0.0 --port 8001 & streamlit run src/app.py --server.port 8080 --server.address 0.0.0.0"]
