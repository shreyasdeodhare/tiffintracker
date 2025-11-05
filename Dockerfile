FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt


ENV FLASK_APP=app.py

RUN mkdir -p /app/data

EXPOSE  5000

CMD ["flask" ,"run","--host=0.0.0.0","--port=5000"]