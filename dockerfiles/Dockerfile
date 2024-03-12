ARG pyver=3.10
FROM python:${pyver}-slim-bookworm

# shiva
RUN apt-get update

# install shiva requirements (Examples, etc..)
COPY ./dockerfiles/docker_requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip wheel && python -m pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

# copy shiva code
RUN mkdir -p /shiva/shiva
RUN mkdir -p /shiva/examples
COPY ./pyproject.toml /shiva
COPY ./README.md /shiva
COPY ./shiva/. /shiva/shiva
COPY ./examples/. /shiva/examples

# install shiva
RUN python -m pip install -e /shiva

# base entry point
CMD ["bash"]
