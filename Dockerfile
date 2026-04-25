FROM python:3.11-slim

WORKDIR /app

# Install nginx
RUN apt-get update && \
    apt-get install -y nginx && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source code
COPY . .

# Install the package itself
RUN pip install --no-cache-dir -e .

# Copy nginx config
COPY nginx.conf /etc/nginx/nginx.conf

# Create data directories
RUN mkdir -p /app/data/chroma_data

# Make start script executable
RUN chmod +x start.sh

# HF Spaces expects port 7860 by default
EXPOSE 7860

CMD ["./start.sh"]
