FROM alpine:latest

# Add libraries
RUN apk update
RUN apk add --no-cache python3 git linux-headers libffi-dev lapack g++ gfortran
RUN apk add --no-cache fftw-dev libtool automake autoconf make cmake openblas-dev curl-dev cfitsio-dev
RUN apk add --no-cache --repository http://dl-cdn.alpinelinux.org/alpine/edge/testing plplot-dev
RUN apk add --no-cache py3-pip
# MUSL fails installing scipy and numpy using pip,
# therefore install it before pip is executed
RUN apk add py3-scipy py3-scikit-learn py3-matplotlib

# Download sources
WORKDIR /root
RUN git clone https://github.com/astromatic/sextractor.git
RUN git clone https://github.com/astromatic/scamp.git
RUN git clone https://github.com/astromatic/swarp.git
RUN git clone https://github.com/smeingast/vircampype.git

# Install SExtractor
WORKDIR /root/sextractor
RUN ./autogen.sh
RUN ./configure  --enable-openblas --with-openblas-incdir=/usr/include
RUN make 
RUN make install

# Install Scamp
# TODO: Check if plplot works (it does not for now...)
WORKDIR /root/scamp
RUN ./autogen.sh
RUN ./configure --enable-openblas --with-openblas-incdir=/usr/include --enable-plplot --with-plplot-incdir=/usr/include
RUN make
RUN make install

# Install Swarp
WORKDIR /root/swarp
RUN ./autogen.sh
RUN ./configure 
RUN make
RUN make install

# Install pipeline
WORKDIR /root/vircampype
RUN pip install -r requirements.txt
RUN pip install .

# Set alias for pipeline worker
RUN ln -s /root/vircampype/vircampype/pipeline/worker.py /usr/bin/vircampype

# Clean up
RUN rm -rf /root/sextractor /root/scamp /root/swarp /root/vircampype/.git && \
    apk del git build-base gfortran automake autoconf cmake && \
    rm -rf /var/cache/apk/*