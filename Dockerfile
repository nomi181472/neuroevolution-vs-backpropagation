# Stage 1: Base stage with the first set of dependencies
FROM python:3.10.15

# Set the working directory
WORKDIR /usr/src/app

# Update package list and install swig
RUN apt-get update && apt-get install -y swig

# Upgrade pip
RUN pip install --upgrade pip

# Copy the first requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt --verbose

# Copy the application code
COPY . .

# Define the command to run the application
CMD ["python", "./cosyne_neuro.py"]
