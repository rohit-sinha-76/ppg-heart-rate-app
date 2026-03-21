# Use a lightweight Python image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Ensure logs are flushed immediate
ENV PYTHONUNBUFFERED=1

# Install system dependencies if required (e.g., for numerical ops support)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything else (Respecting .dockerignore)
COPY . .

# Expose local endpoint trigger handles
EXPOSE 5000

# Run the app binding correctly
CMD ["python", "app.py"]
