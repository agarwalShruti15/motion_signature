# Ubuntu base for building things with gcc, cmake and the like

FROM ubuntu:18.04 as base_ubuntu

RUN     apt-get update -qq \
    &&  apt-get install -qq \
            curl \
            apt-transport-https \
            ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN     apt-get update -qq \
    &&  apt-get install -y --no-install-recommends \
            arp-scan \
            build-essential \
            git \
            iproute2 \
            iputils-ping \
            dnsutils \
            make \
            ninja-build \
            pkg-config \
            software-properties-common \
    && rm -rf /var/lib/apt/lists/*

RUN     apt-get update -qq \
    &&  apt-get install -y --no-install-recommends \
            lsb-release \
            python3-dev \
            python3-setuptools \
            sqlite3 \
            ssh \
            sudo \
            tmux \
            unzip \
            wget \
            zsh \
    && rm -rf /var/lib/apt/lists/*

RUN     wget https://golang.org/dl/go1.14.5.linux-amd64.tar.gz -O /tmp/go.tar.gz \
    &&  tar -xzf /tmp/go.tar.gz -C /usr/local/ \
    &&  rm -rf /tmp/go.tar.gz

ENV RUSTUP_HOME=/opt/rust \
    CARGO_HOME=/opt/rust \
    PATH=/opt/rust/bin:$PATH
RUN     curl -sSfL sh.rustup.rs | sh -s -- -y
RUN     echo $CARGO_HOME \
    &&  cargo install  exa zoxide b3sum shishua_rs rm-improved ripgrep

RUN exa /

RUN curl --silent --show-error \
    https://bootstrap.pypa.io/get-pip.py | python3

RUN mkdir -p /src \
    &&  git clone https://github.com/xkortex/bashbox.git /src/bashbox

RUN bash -c "$(curl -fsSL https://raw.githubusercontent.com/xkortex/bashbox/master/uber-setup.sh)"

RUN     groupadd -g 777 share \
    &&  useradd --uid=777 share \
    &&  chmod -R 775 /src \
    &&  chown -R /src

## =================================================================

FROM base_ubuntu as base_python

WORKDIR /src/motion_signature
COPY ./better_requirements.txt /src/motion_signature/better_requirements.txt

# dlib is not going to work right now
RUN pip install --no-cache-dir \
        numpy \
        torch~=1.1 \
        torchvision~=0.3

RUN pip install --no-cache-dir -r /src/motion_signature/better_requirements.txt

# =================================================================
FROM base_python as openface
COPY --from=gitlab-registry.mediforprogram.com/kitware/berkeley-poi/base/openface \
    /root/diff /

# =================================================================
FROM openface as app

COPY . /src/motion_signature
