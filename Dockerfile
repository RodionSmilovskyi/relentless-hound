# syntax=docker/dockerfile:1

FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/tensorflow-training:2.14.1-gpu-py310-cu118-ubuntu20.04-ec2

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

