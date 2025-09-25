# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Installer dépendances système utiles (si besoin)
RUN apt-get update && apt-get install -y --no-install-recommends gcc libatlas-base-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
