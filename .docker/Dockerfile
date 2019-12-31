FROM continuumio/miniconda3:4.7.12

ENTRYPOINT ["/bin/bash"]

RUN \
    # Add required packages for extension building
    apt update -y && \
    apt install -y gcc 
RUN \
    # Set up base conda
    conda init bash && \
    . /opt/conda/etc/profile.d/conda.sh && \
    # Update conda
    conda update --yes -n base -c defaults conda && \
    # Set up channel priority
    conda config --prepend channels conda-forge && \
    conda config --prepend channels anaconda && \
    # Read channel priority
    conda config --show channels && \
    # Install testing packages to base
    conda install --yes pytest \
                        tox-conda \
                        setuptools

WORKDIR /tmp
