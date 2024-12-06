# Use an official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY ./src /app/src
COPY ./models /app/models
COPY requirements.txt /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install "numpy<2"
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir transformers[torch] accelerate -U
# Expose port
EXPOSE 8000
EXPOSE 5000

# Run the application
CMD ["python", "src/api.py"]