FROM python:3.9-alpine

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY images_api.py .

RUN mkdir -p /app/images

EXPOSE 80

CMD ["uvicorn", "images_api:app", "--host", "0.0.0.0", "--port", "80"]
