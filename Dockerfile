FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# Copy the environment file and update the base conda environment
COPY environment.yml .
RUN conda env update -n base -f environment.yml

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 8000

# Start the FastAPI server using uvicorn
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]