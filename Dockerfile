# Use an official, lightweight Python version
FROM python:3.11-slim

# Set the working folder inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install -r requirements.txt

# Copy the rest of your app's code into the container
COPY . .

# Tell Docker to run Uvicorn to serve the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]