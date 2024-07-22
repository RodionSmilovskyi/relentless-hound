# syntax=docker/dockerfile:1

FROM tensorflow/tensorflow:2.15.0-gpu

COPY code /opt/program

ENV PATH="/opt/program:${PATH}"

WORKDIR /opt/program

RUN chmod a+x train 

RUN pip install -r docker_requirements.txt

RUN rm -rf output

RUN rm -rf __pycache__

