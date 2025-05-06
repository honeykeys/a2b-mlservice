# Dockerfile (Example for ETL/Prediction Job)

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install OS packages if needed (e.g., for certain ML libraries)
# RUN apt-get update && apt-get install -y --no-install-recommends some-package && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy ALL necessary source code directories
# Adjust these COPY commands based on your project structure!
COPY data_processing/ ./data_processing/
COPY modeling/ ./modeling/
COPY prediction/ ./prediction/
# Copy any other shared utility modules

# Default command (can be overridden in Task Definition)
# Running the prediction script might be a sensible default if it handles ETL checks
CMD ["python", "prediction/generate_predictions.py"]
# Or just provide bash for debugging:
# CMD ["bash"]