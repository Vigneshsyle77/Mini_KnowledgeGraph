# ================================
# AI-KnowMap - Streamlit Dockerfile
# ================================
FROM python:3.10-slim
# Preventing python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create app directory
WORKDIR /app

# Install system dependencies for pyvis / numpy / sklearn
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list first for caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy rest of repository
COPY . /app

# Expose port
EXPOSE 8501

# Streamlit configuration (sets server to allow external connections)
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=false
ENV STREAMLIT_SERVER_ENABLEXSRSFPROTECTION=false

# Run the app
CMD ["streamlit", "run", "modules/app.py", "--server.address=0.0.0.0"]
