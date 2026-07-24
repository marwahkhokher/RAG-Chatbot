# Stage 1: Build the React UI
FROM node:20-slim AS builder
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack enable
WORKDIR /app
COPY UI/package.json UI/pnpm-lock.yaml ./
COPY UI/patches/ ./patches/
RUN pnpm install --frozen-lockfile
COPY UI/ ./
RUN pnpm run build

# Stage 2: Build the FastAPI backend
FROM python:3.11-slim

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    libx11-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY . .

# Copy built React files from stage 1 into a static directory
COPY --from=builder /app/dist/public /code/app/static

EXPOSE 8000

# Render sets $PORT automatically; default to 8000 for local runs
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
