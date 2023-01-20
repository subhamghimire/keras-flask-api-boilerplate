FROM python:3.10-slim-buster

RUN apt update \
    && pip install --upgrade pip 

# copy the requirements file into the image
COPY ./requirement.txt /app/requirement.txt

# switch working directory
WORKDIR /app

# install the dependencies and packages in the requirements file
RUN pip install -r requirement.txt

# copy every content from the local file to the image
COPY . /app

EXPOSE 5000

# configure the container to run in an executed manner
ENTRYPOINT [ "python3" ]

CMD ["app.py", "--host=0.0.0.0"]
