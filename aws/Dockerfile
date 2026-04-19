FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your Python files (simulation + dependencies)
COPY *.py ./

# Set default environment variables
ENV GROQ_API_KEY=""
ENV OUTPUT_BUCKET=""
ENV OUTPUT_PREFIX="simulation-output"
ENV AWS_DEFAULT_REGION="us-east-1"

# Default command - can be overridden in ECS task definition
CMD ["python", "simulation.py", "--ticks", "5", "--verbose"]