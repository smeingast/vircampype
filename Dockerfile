# ---- Stage 1: Builder ----
FROM fedora:43 AS builder

RUN dnf update -y && dnf install -y \
    git automake gcc gcc-c++ libtool wget \
    fftw-devel openblas-devel cfitsio-devel plplot-devel \
    libcurl-devel python3-devel python3.14-pip python3.14-cython \
    wcslib-devel gsl-devel \
    && dnf clean all && rm -rf /var/cache/dnf

RUN ln -sf /usr/bin/python3.14 /usr/bin/python && \
    ln -sf /usr/bin/pip3.14 /usr/bin/pip

WORKDIR /root

# Build all C tools in one layer, then clean sources and strip binaries
RUN git clone -b feature/multi-seeing-class-star https://github.com/smeingast/sextractor.git && \
    cd sextractor && ./autogen.sh && ./configure --enable-openblas && make -j"$(nproc)" && make install && \
    cd /root && git clone https://github.com/astromatic/scamp.git && \
    cd scamp && ./autogen.sh && ./configure --enable-openblas --enable-plplot && make -j"$(nproc)" && make install && \
    cd /root && git clone https://github.com/astromatic/psfex.git && \
    cd psfex && ./autogen.sh && ./configure --enable-openblas --enable-plplot && make -j"$(nproc)" && make install && \
    cd /root && git clone https://github.com/astromatic/skymaker.git && \
    cd skymaker && ./autogen.sh && CFLAGS="-O2 -std=gnu17" ./configure && make -j"$(nproc)" && make install && \
    cd /root && git clone https://github.com/astromatic/swarp.git && \
    cd swarp && ./autogen.sh && ./configure && make -j"$(nproc)" && make install && \
    cd /root && wget -q https://ftp.halifax.rwth-aachen.de/gnu/gnuastro/gnuastro-0.24.tar.gz && \
    tar xfz gnuastro-0.24.tar.gz && rm gnuastro-0.24.tar.gz && \
    cd gnuastro-0.24 && CPPFLAGS="-I/usr/include/cfitsio" ./configure && make -j"$(nproc)" && make install && \
    cd /root && rm -rf sextractor scamp psfex skymaker swarp gnuastro-0.24 && \
    find /usr/local/bin -type f -executable -exec strip {} + 2>/dev/null; \
    find /usr/local/lib -name '*.so*' -exec strip --strip-unneeded {} + 2>/dev/null; \
    ldconfig

# Install Python package
COPY . /root/vircampype
RUN pip install --no-cache-dir /root/vircampype


# ---- Stage 2: Dev (use with: docker build --target=dev .) ----
FROM builder AS dev

RUN dnf install -y python3.14-ipython ImageMagick && \
    git clone https://github.com/granttremblay/eso_fits_tools.git && \
    cd eso_fits_tools && make && cp dfits fitsort /usr/bin/ && \
    cd /root && rm -rf eso_fits_tools && \
    dnf clean all && rm -rf /var/cache/dnf

WORKDIR /root/vircampype
RUN pip install --no-cache-dir -e .
WORKDIR /home


# ---- Stage 3: Runtime (default) ----
FROM fedora:43

RUN dnf install -y --setopt=install_weak_deps=False \
    fftw-libs openblas-threads cfitsio plplot libcurl wcslib gsl \
    python3.14 fpack rsync procps-ng \
    && dnf clean all && rm -rf /var/cache/dnf/* /tmp/* /var/tmp/*

RUN ln -sf /usr/bin/python3.14 /usr/bin/python

# Copy compiled C tools, shared libraries, and Python packages from builder
COPY --from=builder /usr/local/ /usr/local/
RUN ldconfig

# Create entry-point wrapper
RUN printf '#!/usr/bin/env python\nfrom vircampype.pipeline.worker import main\nraise SystemExit(main())\n' \
    > /usr/bin/vircampype && chmod +x /usr/bin/vircampype

WORKDIR /home
