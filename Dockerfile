FROM node:22-alpine AS frontend-build

WORKDIR /frontend
COPY app/frontend/package*.json ./
RUN npm ci
COPY app/frontend/ ./
RUN npm run build

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt

RUN groupadd --system app \
    && useradd --system --gid app --create-home app \
    && mkdir -p /app/state \
    && chown app:app /app/state

COPY --chown=app:app app/main.py app/auth.py ./app/
COPY --from=frontend-build --chown=app:app /frontend/dist ./app/frontend/dist

USER app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
