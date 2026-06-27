FROM python:3.12-slim

WORKDIR /app

# system deps kept minimal; scipy/numpy ship wheels
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# OPENROUTER_API_KEY is injected at runtime by the host's secrets (never baked in)
ENV PORT=8000
EXPOSE 8000

# single server: numerics + secure LLM proxy + static frontend
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
