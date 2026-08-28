# Hugging Face Spaces runs this container. It serves the FastAPI app and the
# one-page demo from a single process on one URL.
FROM python:3.11-slim

# Spaces runs as uid 1000 and only $HOME is writable.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR $HOME/app

# Dependencies first so a code change does not reinstall CatBoost.
COPY --chown=user requirements-serve.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements-serve.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
