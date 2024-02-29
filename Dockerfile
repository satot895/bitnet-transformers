from nvidia/cuda:12.0.0-devel-ubuntu20.04

RUN apt update; \
    apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev wget git libbz2-dev liblzma-dev; \
    wget https://www.python.org/ftp/python/3.10.8/Python-3.10.8.tgz; \
    tar -zxvf Python-3.10.8.tgz; \
    cd Python-3.10.8; \
    ./configure --prefix=/usr/local/python3; \
    make && make install; \
    ln -sf /usr/local/python3/bin/python3.10 /usr/bin/python3; \
    ln -sf /usr/local/python3/bin/pip3.10 /usr/bin/pip

WORKDIR /opt/

RUN python3 -m venv venv
ENV PATH /opt/venv/bin:${PATH}

WORKDIR /opt/

# install packages
# COPY ./ /opt/bitnet/
RUN git clone https://github.com/beomi/bitnet-transformers

WORKDIR /opt/bitnet-transformers/

RUN pip install -r clm_requirements.txt
RUN pip install mlflow

RUN git clone https://github.com/huggingface/transformers
RUN pip install -e transformers

RUN rm ./transformers/src/transformers/models/llama/modeling_llama.py
RUN ln -s $(pwd)/bitnet_llama/modeling_llama.py ./transformers/src/transformers/models/llama/modeling_llama.py


COPY ./run_clm.py /opt/bitnet-transformers/
COPY ./train_wikitext.sh /opt/bitnet-transformers/
COPY ./.env /opt/bitnet-transformers/


# RUN pip install -r /opt/bitnet/clm_requirements.txt && \
#     rm -r ~/.cache/pip

