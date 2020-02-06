FROM python:3.7-buster

#RUN apk upgrade --update && apk add --no-cache python3 python3-dev gcc gfortran freetype-dev musl-dev libpng-dev g++ openblas-dev py-numpy@community

#RUN pip3 install virtualenv; pip3 install --upgrade pip;


ADD ./requirements.txt /tmp/

RUN pip3 --version; pip3 install --user --requirement /tmp/requirements.txt