# Use a small Python base
FROM python:3.11-slim

# System deps (keep minimal; SciPy wheels avoid compilers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . /app

# Streamlit needs a writable cache dir in some environments
ENV STREAMLIT_CACHE_DIR=/tmp/streamlit-cache
ENV PYTHONUNBUFFERED=1

# Expose is optional in Vercel, but nice for local run
EXPOSE 7860

# IMPORTANT: Vercel provides $PORT. Use sh -c so $PORT expands.
CMD ["sh", "-c", "streamlit run app.py --server.port $PORT --server.address 0.0.0.0"]
