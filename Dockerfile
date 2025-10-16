FROM nvcr.io/nvidia/pytorch:23.09-py3
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && \
    apt-get install -y git-lfs && \
    git lfs install
COPY . .
CMD ["python", "scripts/pipeline_batch.py"]