# Container for the scoring service. Serves the API and the one-page demo from a
# single process on one URL.
#
# Build context is the repository root, not this directory:
#     docker build -f deploy/Dockerfile .
#
# Python 3.13 deliberately matches the interpreter the model artifact was
# pickled with. The pinned library versions are identical to requirements.txt,
# so the saved pipeline loads without a cross-version unpickling risk.
FROM python:3.13-slim

# Run as a non-root user. Render and most container hosts prefer this, and it
# keeps $HOME writable for the pip user install.
RUN useradd -m -u 1000 appuser
USER appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR $HOME/app

# Dependencies first, so editing application code does not reinstall CatBoost.
COPY --chown=appuser deploy/requirements-serve.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir --only-binary=:all: -r requirements-serve.txt

COPY --chown=appuser . .

# Render injects $PORT and expects the process to bind it. The default keeps the
# image runnable locally and on hosts that do not set it.
ENV PORT=7860
EXPOSE 7860

# Shell form so $PORT expands at runtime rather than being taken literally.
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}
