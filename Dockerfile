from alpine:latest

RUN apk update
RUN apk add python3 git
# MUSL fails installing scipy and numpy using pip,
# therefore install it before pip is executed
RUN apk add lapack g++ gfortran
RUN apk add py3-scipy py3-scikit-learn py3-matplotlib
RUN apk add py3-pip 



# COMMENT THIS OUT LATER
COPY . /usr/src/app/

WORKDIR /usr/src/app

# REPLACE BY GIT CLONE
#RUN git clone bla .


RUN pip install -r requirements.txt
RUN python setup.py build
RUN python setup.py install
RUN pip install vircampype

