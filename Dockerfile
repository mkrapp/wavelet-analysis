# start by pulling the python image
FROM python:slim

# copy the requirements file into the image
COPY ./requirements-docker.txt /app/requirements.txt

# switch working directory
WORKDIR /app

#RUN apk add --no-cache --update \
#    python3 python3-dev gcc g++ \
#    gfortran musl-dev lapack-dev openblas-dev

# install the dependencies and packages in the requirements file
RUN pip install -r requirements.txt

# copy every content from the local file to the image
COPY . /app

# configure the container to run in an executed manner
CMD ["gunicorn" , "-b", "0.0.0.0", "main:app"]
