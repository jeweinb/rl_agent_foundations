FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir gymnasium numpy dash plotly scipy scikit-learn

# Copy application code (production modules only)
COPY config.py .
COPY environment/ environment/
COPY models/ models/
COPY training/ training/
COPY simulation/metrics.py simulation/metrics.py
COPY simulation/logger.py simulation/logger.py
COPY dashboard/ dashboard/
COPY deploy/inference_server.py .

# Copy model checkpoint (mounted or baked in)
# In production: mount from S3/GCS via volume
COPY training/checkpoints/ checkpoints/

EXPOSE 8080

CMD ["python", "inference_server.py", "--port", "8080"]
