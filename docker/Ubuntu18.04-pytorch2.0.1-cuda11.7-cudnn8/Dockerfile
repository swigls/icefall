FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

# ENV http_proxy=http://aaa.bbb.cc.net:8080 \
#	https_proxy=http://aaa.bbb.cc.net:8080

# install normal source
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        g++ \
        make \
        automake \
        autoconf \
        bzip2 \
        unzip \
        wget \
        sox \
        libtool \
        git \
        subversion \
        zlib1g-dev \
        gfortran \
        ca-certificates \
        patch \
        ffmpeg \
        valgrind \
		libssl-dev \
	    vim \
		curl

# cmake
RUN wget -P /opt https://cmake.org/files/v3.18/cmake-3.18.0.tar.gz && \
    cd /opt && \
    tar -zxvf cmake-3.18.0.tar.gz && \
    cd cmake-3.18.0 && \
    ./bootstrap && \
    make && \
    make install && \
    rm -rf cmake-3.18.0.tar.gz && \
    find /opt/cmake-3.18.0 -type f \( -name "*.o" -o -name "*.la" -o -name "*.a" \) -exec rm {} \; && \
    cd -

# flac
RUN wget -P /opt https://downloads.xiph.org/releases/flac/flac-1.3.2.tar.xz  && \
    cd /opt && \
    xz -d flac-1.3.2.tar.xz && \
    tar -xvf flac-1.3.2.tar && \
    cd flac-1.3.2 && \
    ./configure && \
    make && make install && \
    rm -rf flac-1.3.2.tar && \
    find /opt/flac-1.3.2  -type f \( -name "*.o" -o -name "*.la" -o -name "*.a" \) -exec rm {} \; && \
    cd -

#RUN conda install -y -c pytorch torchaudio=0.12 && \
RUN pip install graphviz


#install k2 from source
RUN git clone https://github.com/k2-fsa/k2.git /opt/k2 && \
    cd /opt/k2 && \
    python3 setup.py install && \
    cd -
#    export K2_MAKE_ARGS="-j6" && \
#

# install  lhotse
RUN cd /opt
RUN pip install git+https://github.com/lhotse-speech/lhotse

RUN git clone https://github.com/k2-fsa/icefall /opt/icefall && \
	cd /opt/icefall && \
	pip install -r requirements.txt

RUN pip install kaldifeat
ENV PYTHONPATH /opt/icefall:$PYTHONPATH

RUN pip install gin-config

# WORKDIR /workspace/icefall
