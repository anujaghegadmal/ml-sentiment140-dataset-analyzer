# Using an official Python runtime as a parent image
FROM python:3.9-slim

# Setting the working directory in the container
WORKDIR /app

# Copying requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying the rest of the application code
COPY . .

# Exposing a port (optional, if the application runs as a service)
EXPOSE 8000

# Defining the command to run the application
CMD ["python", "app.py"]
