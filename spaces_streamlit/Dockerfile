FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src ./src
COPY .streamlit ./.streamlit

# Create data directory for SQLite
RUN mkdir -p /data

# Expose Streamlit port
EXPOSE 7860

ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV FAKESCOPE_DB_PATH=/data/predictions.db

# Run Streamlit
CMD ["streamlit", "run", "src/app.py", "--server.port=7860", "--server.address=0.0.0.0"]
