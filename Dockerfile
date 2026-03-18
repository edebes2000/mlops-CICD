# --------------------------------------------------------
# Serving container only
# Packages the FastAPI inference service, NOT training code.
# --------------------------------------------------------

FROM continuumio/miniconda3:latest

# Set the working directory
WORKDIR /app

# This ensures that logs are flushed immediately, which is crucial
# for real-time monitoring and debugging in production environments like Render
ENV PYTHONUNBUFFERED=1

# This is important in a containerized environment where we want to minimize
# unnecessary file writes and ensure that the application runs with the latest
# code without worrying about stale bytecode files. It also helps to reduce the
# image size by not generating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure Python can resolve absolute imports like `src.api`
# Crucial for the FastAPI application to run correctly, especially when using a
# modular project structure
ENV PYTHONPATH=/app

# Copy the pre-compiled lockfile first to leverage Docker layer caching.
# This file was generated locally using: conda-lock -p linux-64 -f environment.yml
COPY conda-lock.yml .

# 1. Install conda-lock itself into the container
# 2. Use conda-lock to install the exact frozen dependencies into the base environment
# 3. Clean the cache to reduce image size
RUN conda install -c conda-forge conda-lock -y && \
    conda-lock install -n mlops conda-lock.yml && \
    conda clean -afy

# We set the PATH to include the mlops environment so that when we run commands like `uvicorn`,
# it uses the correct Python environment with all dependencies installed
ENV PATH=/opt/conda/envs/mlops/bin:$PATH

# Copy the rest of the application code
# .dockerignore used to exclude unnecessary files (e.g., .git, __pycache__, etc.)
COPY . .

# Expose a default port (mostly for local documentation)
EXPOSE 8000

# Tell Docker/Render how to test if the API is actually healthy
# Crucial for production deployments to ensure that the service is running
# and responsive before accepting traffic. It helps with auto-scaling and load balancing.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8000}/health || exit 1

# Start the FastAPI server using uvicorn
# We use 'sh -c' to allow environment variable expansion for Render's dynamic $PORT
CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT:-8000}"]