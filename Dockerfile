FROM fedora:43
ARG BUILD_OPTION=user

# Install libraries
RUN dnf update -y && dnf install -y git automake gcc gcc-c++ libtool fftw-devel openblas-devel \
    cfitsio-devel plplot-devel libcurl-devel python3-devel python3.14-pip python3.14-cython \
    wcslib-devel gsl-devel wget fpack

# Set Python aliases
RUN ln -sf /usr/bin/python3.14 /usr/bin/python && \
    ln -sf /usr/bin/pip3.14 /usr/bin/pip

# Set workdir
WORKDIR /root

# Install more stuff for dev mode
RUN if [ "$BUILD_OPTION" = "dev" ] ; then \
    dnf install -y python3.14-ipython ImageMagick && \
    git clone https://github.com/granttremblay/eso_fits_tools.git && \
    cd eso_fits_tools && make && cp dfits fitsort /usr/bin/ ; \
    fi

# Install SExtractor
RUN git clone https://github.com/astromatic/sextractor.git && \
    cd /root/sextractor && \
    ./autogen.sh && ./configure --enable-openblas && make && make install

# Install Scamp
RUN git clone https://github.com/astromatic/scamp.git && \
    cd /root/scamp && \
    ./autogen.sh && ./configure --enable-openblas --enable-plplot && make && make install

# Install Swarp
RUN git clone https://github.com/astromatic/swarp.git && \
    cd /root/swarp && \
    ./autogen.sh && ./configure && make && make install

# Install gnuastro (main: https://ftp.gnu.org/gnu/gnuastro/gnuastro-0.24.tar.gz)
RUN wget https://ftp.halifax.rwth-aachen.de/gnu/gnuastro/gnuastro-0.24.tar.gz && \
    tar xfz gnuastro-0.24.tar.gz && \
    rm -f gnuastro-0.24.tar.gz && \
    cd gnuastro-0.24 && \
    export CPPFLAGS="-I/usr/include/cfitsio" && \
    ./configure && make && make install

# Install requirements
COPY requirements.txt /root/vircampype/
WORKDIR /root/vircampype
RUN pip install -r requirements.txt

# Copy pipeline
COPY . /root/vircampype

# Install pipeline
RUN if [ "$BUILD_OPTION" = "dev" ] ; then \
        pip install -e . ; \
    elif [ "$BUILD_OPTION" = "user" ] ; then \
        pip install . && \
        rm -rf /root/scamp /root/sextractor /root/swarp /root/gnuastro-0.24 && \
        dnf clean all && rm -rf /var/cache/dnf/* /tmp/* /var/tmp/* && \
        ln -s /root/vircampype/vircampype/pipeline/worker.py /usr/bin/vircampype ; \
    fi

# Set working directory
WORKDIR /home