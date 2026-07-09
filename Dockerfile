# CPU-only image -- matches HF Spaces free tier. Installs the CPU torch
# wheel explicitly (smaller, no CUDA deps) before the rest of requirements.txt.
FROM python:3.11-slim

WORKDIR /code

# curl is needed by entrypoint.sh's health-check polling loop
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY ui ./ui
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# HF Spaces containers run as a non-root user; the HF cache dir must be
# writable or every checkpoint/tokenizer download will fail at runtime.
ENV HF_HOME=/code/.cache/huggingface
RUN mkdir -p /code/.cache/huggingface && chmod -R 777 /code/.cache

EXPOSE 7860
ENV GRADIO_PORT=7860
ENV API_PORT=8000

CMD ["./entrypoint.sh"]
