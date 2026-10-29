FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml Readme.md ./
COPY configs ./configs
COPY src ./src
COPY UI ./UI

RUN pip install --upgrade pip \
    && pip install -e ".[api,ui]"

EXPOSE 8001 7860

