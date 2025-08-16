FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

# Copy application files
COPY enhanced_arbitrage_simulator.py .
COPY cleaned_data/ ./cleaned_data/
COPY *.md ./

# Create streamlit config directory
RUN mkdir -p ~/.streamlit

# Create streamlit config
RUN echo '[server]\n\
headless = true\n\
port = 8501\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
' > ~/.streamlit/config.toml

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
CMD ["streamlit", "run", "enhanced_arbitrage_simulator.py", "--server.port=8501", "--server.address=0.0.0.0"]