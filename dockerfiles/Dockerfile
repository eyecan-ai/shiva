ARG pyver=3.10
FROM python:${pyver}-slim-bookworm

# opencv requirements
RUN apt-get update && apt-get install -y git

RUN python -m pip install --no-cache-dir --upgrade pip

WORKDIR home/

RUN git clone https://github.com/eyecan-ai/shiva.git

RUN python -m pip install --no-cache-dir shiva/. 

# For examples
RUN python -m pip install requests

# base entry point
CMD ["bash"]
