# This is the default base image for use with user models and workflows.
FROM ubuntu:18.04
ENV LC_ALL=en_US.UTF-8 TERM=xterm COLS=132 ROWS=43 DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y locales && locale-gen $LC_ALL && \
    apt-get install --no-install-recommends -y \
            python3-pip \
            python3-dev \
            openjdk-11-jdk \
            openjdk-11-jre \
            gcc \
            libc6-dev \
            build-essential \
            maven \
            apt-utils \
            curl \
            gpg-agent \
            software-properties-common \
            dirmngr \
            libssl-dev \
            ca-certificates \
            locales \
            libcurl4-openssl-dev \
            libxml2-dev \
            libgomp1 \
            pandoc \
            nginx \
            && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    chmod 707 /var/lib/nginx

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

RUN pip3 install -U pip
RUN pip3 install --no-cache-dir setuptools wheel


# do the trick to install drum deps
RUN pip3 install --no-cache-dir datarobot-drum
RUN pip3 uninstall -y datarobot-drum
