FROM python:3.10.16-slim

WORKDIR /app

# Copy deps first for better build caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app & model
COPY src/advertising_ml.py .


CMD ["python","advertising_ml.py" ]
 
