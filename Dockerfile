FROM python:3.8-slim

WORKDIR /app

COPY . /app

RUN apt update -y && apt install -y gcc build-essential

RUN pip install --upgrade pip && pip install -r requirements.txt

CMD ["streamlit", "run", "app.py", "--server.port=80", "--server.address=0.0.0.0"]
