FROM python:3.12-slim-bookworm
# Set the working directory in the container
WORKDIR /app

# Copy the entire application code into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -U langchain-openai

# Expose the FastAPI default port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]