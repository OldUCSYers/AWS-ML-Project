# FROM python:3.10-slim

# WORKDIR /app

# # deps (paths are relative to the BUILD CONTEXT = repo root)
# COPY infrastructure/requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # copy the whole app folder
# COPY src/ ./src

# # run your script
# CMD ["python", "src/advertising_ml.py"]

# Python 3.11 Lambda base (includes Runtime Interface + right entrypoint)
FROM public.ecr.aws/lambda/python:3.11

# Copy code and deps
 COPY infrastructure/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -t ${LAMBDA_TASK_ROOT}

COPY  src/ ${LAMBDA_TASK_ROOT}

# Tell Lambda your handler: <module>.<function>
CMD ["advertising_ml.lambda_handler"]

