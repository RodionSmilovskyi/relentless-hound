# syntax=docker/dockerfile:1

FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

COPY package /opt/package

ENV PYTHONUNBUFFERED=TRUE

ENV PYTHONDONTWRITEBYTECODE=TRUE

ENV WRAPT_DISABLE_EXTENSIONS=TRUE

ENV PATH="/opt/package:${PATH}"

ENV PYTHONPATH="/opt"

WORKDIR /opt/package

RUN chmod a+x train 

RUN pip install -r docker_requirements.txt

RUN rm -rf output

RUN rm -rf __pycache__

