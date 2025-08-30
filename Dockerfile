# Use Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (to use Docker cache efficiently)
COPY requirements.txt .

# Install dependencies (force compatible numpy for spaCy + thinc)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt && \
    python -m spacy download en_core_web_sm

# Copy project files
COPY . .

# Expose port
EXPOSE 8000

# Start Gunicorn server
CMD ["gunicorn", "--bind", "0.0.0.0:${PORT}", "app:app"]

