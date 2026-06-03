FROM python:3.12-slim

WORKDIR /app

# install system dependencies needed by LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# create runtime directories
RUN mkdir -p /app/logs /app/input /app/output /app/models && \
    touch /app/logs/service.log && \
    chmod -R 777 /app/logs /app/input /app/output

# install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy source code and model artifact
COPY . .

# mount points for input and output data
VOLUME /app/input
VOLUME /app/output

CMD ["python", "-u", "./app/app.py"]
