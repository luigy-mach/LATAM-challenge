# syntax=docker/dockerfile:1.2
FROM python:3.12.3-slim-bookworm
# put you docker configuration here

# working directory
WORKDIR /app

# copy only requirements files
COPY requirements.txt .
COPY requirements-dev.txt .
COPY requirements-test.txt .

# install requirements
RUN pip install -r requirements.txt -r requirements-dev.txt -r requirements-test.txt

# # user no-root
# RUN useradd -m appuser && chown -R appuser:appuser /app
# USER appuser

# copy all files
COPY . .

# port
EXPOSE 8080

# run the application
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]