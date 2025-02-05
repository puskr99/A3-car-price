# Use the official Python image as a base
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the app folder into the container
COPY app /app

# Install the required Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask app's default port
EXPOSE 5000

# Set the environment variable for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Command to run the Flask application
CMD ["flask", "run"]