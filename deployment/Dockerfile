# Use a slim Python image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the core parsing logic
COPY src/core_parser.py .

# The component is executed via KFP, so no explicit CMD needed here.
# KFP will define the command when the component is called.