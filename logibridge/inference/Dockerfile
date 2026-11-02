# LogiEdge inference service.
# Build (from repo root):
#   docker build --build-context pipeline=data_pipeline -t logibridge-inference:latest inference
# Layer order = least to most frequently changed, so an OTA model update rebuilds only the last layer.
FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir numpy==1.26.4 paho-mqtt==2.1.0 tflite-runtime==2.14.0

COPY --from=pipeline preprocessing.py training_stats.npy ./
COPY inference_service.py .
COPY model.tflite .

ENV MODEL_PATH=/app/model.tflite \
    TRUCK_ID=TRK001 \
    MQTT_HOST=localhost \
    MQTT_PORT=1883 \
    ALERT_LOG=/app/alerts.jsonl \
    PYTHONUNBUFFERED=1

CMD ["python", "inference_service.py"]
