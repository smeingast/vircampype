FROM fedora:37
ARG BUILD_OPTION=user

# Install libraries
RUN dnf install -y git automake gcc gcc-c++ libtool fftw-devel \
    openblas-devel cfitsio-devel plplot-devel libcurl-devel \
    python3-devel python3-pip python3-cython

# Set Python alias
RUN ln -s /usr/bin/python3 /usr/bin/python


# Install more stuff for dev mode
RUN if [ "$BUILD_OPTION" = "dev" ] ; then \
        dnf install -y python3-ipython ImageMagick ; \
    fi

# Download sources
WORKDIR /root
RUN git clone https://github.com/astromatic/sextractor.git && \
    git clone https://github.com/astromatic/scamp.git && \
    git clone https://github.com/astromatic/swarp.git

# Install SExtractor
WORKDIR /root/sextractor
RUN ./autogen.sh
RUN ./configure --enable-openblas
RUN make
RUN make install

# Install Scamp
WORKDIR /root/scamp
RUN ./autogen.sh
RUN ./configure --enable-openblas --enable-plplot
RUN make
RUN make install

# Install Swarp
WORKDIR /root/swarp
RUN ./autogen.sh
RUN ./configure
RUN make
RUN make install

# Copy pipeline
COPY . /root/vircampype
WORKDIR /root/vircampype

# Install requirements
RUN pip install -r requirements.txt

# Only install pipeline as user
RUN if [ "$BUILD_OPTION" = "user" ] ; then \
        pip install . && \
        rm -rf /root/scamp /root/sextractor /root/swarp && \
        dnf clean all && rm -rf /var/cache/dnf/* /tmp/* /var/tmp/ && \
        dnf remove -y git automake gcc gcc-c++ libtool python3-pip && \
        ln -s /root/vircampype/vircampype/pipeline/worker.py /usr/bin/vircampype; \
    fi

# Set working directory
WORKDIR /home
