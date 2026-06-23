# Use x86_64 Python 3.11 slim image
FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /app

# Copy all project files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Expose port for local testing (not used on Render)
EXPOSE 10000

# Start gunicorn server using the Render-provided port
CMD gunicorn server.app:app --bind 0.0.0.0:$PORT
