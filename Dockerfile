FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
RUN mkdir -p videos

CMD ["uvicorn", "app.py:app", "--host", "0.0.0.0", "--port", "8000"]am