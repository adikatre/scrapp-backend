# ------------------------------------------------------------
# Base image: slim Python for smaller image size
# ------------------------------------------------------------
FROM python:3.12-slim

# ------------------------------------------------------------
# System deps needed by opencv-python wheels at runtime
# (no GUI; just enough to load/process images)
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Workdir and app code
# ------------------------------------------------------------
WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py

# ------------------------------------------------------------
# Python deps
# Install from requirements.txt and add torch CPU version
# ------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
      "torch==2.3.1" --index-url https://download.pytorch.org/whl/cpu
    
# ------------------------------------------------------------
# Env (Flask listens on all interfaces inside container)
# ------------------------------------------------------------
ENV PYTHONUNBUFFERED=1
ENV PORT=5000

# ------------------------------------------------------------
# Expose Flask port and set default command
# ------------------------------------------------------------
EXPOSE 5000
CMD ["python", "app.py"]