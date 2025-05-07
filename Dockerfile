# Dockerfile (for the ETL/Prediction Fargate Task)

# Use a standard slim Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install OS packages if any Python libraries require them (e.g., build tools)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     gcc \
#  && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL necessary Python source code directories
# Ensure paths are relative to the Dockerfile (your project root)
COPY data_processing/ ./data_processing/
COPY scheduling/ ./scheduling/
# If generate_predictions.py was in a 'prediction/' folder and is called by run_scheduled_etl.py, copy it too:
# COPY prediction/ ./prediction/
# Copy any other shared utility modules or model files if they were NOT meant to be loaded from S3 for some reason

# Set the default command (this will be overridden in the ECS Task Definition,
# but it's good to have a sensible default or one for testing)
CMD ["python", "scheduling/run_scheduled_etl.py"]