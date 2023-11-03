FROM fedora:37
ARG BUILD_OPTION=user

# Install libraries
RUN dnf install -y git automake gcc gcc-c++ libtool fftw-devel openblas-devel \
    cfitsio-devel plplot-devel libcurl-devel python3-devel python3-pip python3-cython \
    wcslib-devel cfitsio-devel gsl-devel wget fpack

# Set Python alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set workdir
WORKDIR /root

# Install more stuff for dev mode
RUN if [ "$BUILD_OPTION" = "dev" ] ; then  \
    dnf install -y python3-ipython ImageMagick &&  \
    git clone https://github.com/granttremblay/eso_fits_tools.git &&  \
    cd eso_fits_tools && make && cp dfits fitsort /usr/bin/ ; \
    fi

# Download sources for astromatic tools
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

# Install gnuastro
WORKDIR /root/gnuastro
RUN wget https://ftp.gnu.org/gnu/gnuastro/gnuastro-0.21.tar.gz \
    && tar xfz gnuastro-0.21.tar.gz \
    && rm -rf gnuastro-0.21.tar.gz
WORKDIR /root/gnuastro/gnuastro-0.21
ENV CPPFLAGS="$CPPFLAGS -I/usr/include/cfitsio"
RUN ./configure \
    && make -j$(nproc) \
    && make install

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
        ln -s /root/vircampype/vircampype/pipeline/worker.py /usr/bin/vircampype ; \
    fi

# Set working directory
WORKDIR /home
