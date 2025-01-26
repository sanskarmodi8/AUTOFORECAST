FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app

RUN apt update -y && apt install -y && pip install -r requirements.txt 

CMD ["streamlit", "run", "app.py"]