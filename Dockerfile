FROM docker.io/python:3.12

# Prevents Python from writing pyc files.
ENV PYTHONDONTWRITEBYTECODE=1

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y python3 python3-pip git
COPY . /
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn


WORKDIR /app

ENV GUNICORN_CMD_ARGS="--workers=3 --bind=0.0.0.0:8000"
# First, copy the requirements file into the container.

# Download dependencies as a separate step to take advantage of Docker's caching.
# Leverage a cache mount to /root/.cache/pip to speed up subsequent builds.

# Switch to the non-privileged user to run the application.
EXPOSE 8000

# Copy the source code into the container.
COPY . .

# Expose the port that the application listens on.
EXPOSE 8000

# Run the application.
CMD ["gunicorn", "final.app:app", "--bind", "0.0.0.0:8000"]
