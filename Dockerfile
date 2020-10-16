FROM python:3.7

RUN mkdir -p /code/curiosity
RUN mkdir /code/curiosity/dialog_data
RUN mkdir /code/curiosity/models
RUN mkdir /code/curiosity/experiments
WORKDIR /code/curiosity
VOLUME /code/curiosity/dialog_data
VOLUME /code/curiosity/models
VOLUME /code/curiosity/experiments

RUN pip install poetry==1.1.1
COPY pyproject.toml poetry.lock /code/curiosity/
RUN poetry export --without-hashes -f requirements.txt > reqs.txt \
    && pip install -r reqs.txt


CMD ["allennlp"]