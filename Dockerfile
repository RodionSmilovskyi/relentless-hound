# syntax=docker/dockerfile:1

FROM tensorflow/tensorflow:2.15.0-gpu

COPY package /opt/package

ENV PYTHONUNBUFFERED=TRUE

ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV PATH="/opt/package:${PATH}"

ENV PYTHONPATH="/opt"

WORKDIR /opt/package

RUN chmod a+x train 

RUN pip install -r docker_requirements.txt

RUN rm -rf output

RUN rm -rf __pycache__

