FROM --platform=linux/amd64 python:3.7-slim-bullseye as deps-image

RUN \
  apt-get -y update && \
  apt-get --fix-broken -y install && \
  apt-get -y install --no-install-recommends build-essential gcc git

ENV PATH="/opt/venv/covid-data-model/bin:${PATH}"

WORKDIR /covid-data-model
COPY requirements.txt .
COPY setup.py .

RUN \
  python3 -m venv /opt/venv/covid-data-model && \
  . /opt/venv/covid-data-model/bin/activate && \
  pip3 install -r requirements.txt
  
COPY . .

RUN pip3 install .
