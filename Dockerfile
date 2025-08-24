FROM docker.io/python:3.12-slim

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

COPY . .
RUN pip install --no-cache-dir -r requirements.txt


WORKDIR /app

# First, copy the requirements file into the container.

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.

# Switch to the non-privileged user to run the application.
EXPOSE 5000

# Run the application.
CMD ["python", "final/app.py"]