FROM python:3.9.15-slim-bullseye


WORKDIR /app

ADD . /app

COPY requirements.txt /app

RUN pip install -r requirements.txt \
    && rm requirements.txt

COPY . /app

EXPOSE 5000

CMD ["python", "app.py"]