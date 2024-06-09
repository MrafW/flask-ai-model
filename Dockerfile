# Use a Python 3.9 base image
FROM python:3.9

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app

# Create a working directory
WORKDIR $APP_HOME

# Copy the requirements file into the container
COPY requirements.txt ./

# Install OpenGL development packages
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the service account JSON file
COPY da-kalbe-63ee33c9cdbb.json /app/key.json

# Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/key.json

# Copy the rest of the application code
COPY . .

# Expose the port
EXPOSE 8080

# Command to run when the container starts
CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:app"]