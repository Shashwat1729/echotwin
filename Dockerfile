# EchoTwin CPU image: core flows (ingest/dataset/profile/eval/UI) work out of
# the box. Training needs a GPU image; voice synthesis needs backend weights
# mounted or downloaded explicitly at runtime (never baked in).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ECHOTWIN_HOME=/home/echotwin/.echotwin

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY examples/ ./examples/
COPY scripts/ ./scripts/

RUN pip install --no-cache-dir "uv>=0.5" \
 && uv pip install --system -e ".[voice,ui,lang]"

RUN useradd -m echotwin && mkdir -p /home/echotwin/.echotwin && chown -R echotwin /app /home/echotwin
USER echotwin

EXPOSE 8765 7860
CMD ["python", "-m", "echotwin.cli", "doctor"]
