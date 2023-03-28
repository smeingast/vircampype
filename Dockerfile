FROM alpine:latest

# Add libraries
RUN apk update && \
    apk add --no-cache \
        python3 \
        git \
        linux-headers \
        libffi-dev \
        lapack \
        g++ \
        gfortran \
        fftw-dev \
        libtool \
        automake \
        autoconf \
        make \
        cmake \
        openblas-dev \
        curl-dev \
        cfitsio-dev \
        --repository https://dl-cdn.alpinelinux.org/alpine/edge/testing \
        plplot-dev \
        py3-pip \
        py3-scipy \
        py3-scikit-learn \
        py3-matplotlib

# Download sources
WORKDIR /root
RUN git clone https://github.com/astromatic/sextractor.git && \
    git clone https://github.com/astromatic/scamp.git && \
    git clone https://github.com/astromatic/swarp.git && \
    git clone https://github.com/smeingast/vircampype.git

# Install SExtractor, Scamp, and Swarp
RUN apk add --no-cache --virtual .build-deps build-base && \
    cd sextractor && \
    ./autogen.sh && \
    ./configure --enable-openblas --with-openblas-incdir=/usr/include && \
    make && \
    make install && \
    cd ../scamp && \
    ./autogen.sh && \
    ./configure --enable-openblas --with-openblas-incdir=/usr/include --enable-plplot --with-plplot-incdir=/usr/include && \
    make && \
    make install && \
    cd ../swarp && \
    ./autogen.sh && \
    ./configure && \
    make && \
    make install && \
    apk del .build-deps

# Install pipeline
WORKDIR /root/vircampype
RUN pip install -r requirements.txt && \
    pip install .

# Set alias for pipeline worker
RUN ln -s /root/vircampype/vircampype/pipeline/worker.py /usr/bin/vircampype

# Clean up
RUN rm -rf /root/sextractor /root/scamp /root/swarp /root/vircampype/.git && \
    apk del git build-base gfortran automake autoconf cmake && \
    rm -rf /var/cache/apk/*
