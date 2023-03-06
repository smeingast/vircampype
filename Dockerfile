from alpine:latest

RUN apk update
RUN apk add python3 git
# MUSL fails installing scipy and numpy using pip,
# therefore install it before pip is executed
RUN apk add lapack g++ gfortran
RUN apk add py3-scipy py3-scikit-learn py3-matplotlib
RUN apk add py3-pip 


WORKDIR /root
RUN apk add fftw-dev libtool automake autoconf make cmake openblas-dev  curl-dev cfitsio-dev 
RUN git clone https://github.com/astromatic/sextractor.git
RUN git clone https://github.com/astromatic/scamp.git
RUN git clone https://github.com/astromatic/swarp.git
# Install sextractor
WORKDIR /root/sextractor
#RUN /usr/bin/cpufreq-selector -g performance
RUN ./autogen.sh
RUN ./configure  --enable-openblas --with-openblas-incdir=/usr/include
RUN make 
RUN make install
# Install Scamp
#RUN apk add curl-dev
WORKDIR /root/scamp
RUN ./autogen.sh
RUN ./configure --enable-openblas --with-openblas-incdir=/usr/include
RUN make
RUN make install
# Install Swarp
WORKDIR /root/swarp
#RUN apk add cfitsio-dev
RUN ./autogen.sh
RUN ./configure 
RUN make
RUN make install


# COMMENT THIS OUT LATER
COPY . /usr/src/app/

WORKDIR /usr/src/app

# REPLACE BY GIT CLONE
#RUN git clone bla .


RUN pip install -r requirements.txt
RUN python setup.py build
RUN python setup.py install
RUN pip install vircampype

