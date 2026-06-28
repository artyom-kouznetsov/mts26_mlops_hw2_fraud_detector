FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MPLCONFIGDIR=/tmp/matplotlib

# install the system dependency required by lightgbm
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# create runtime directories used by the legacy batch entry point
RUN mkdir -p /app/logs /app/input /app/output /app/models && \
    touch /app/logs/service.log && \
    chmod -R 777 /app/logs /app/input /app/output

# install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source code and the model artifact
COPY . .

CMD ["python", "-u", "./src/kafka_scorer.py"]
