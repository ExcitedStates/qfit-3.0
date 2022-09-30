# See here for image contents: https://github.com/devcontainers/images/blob/main/src/miniconda/.devcontainer/Dockerfile

FROM --platform=linux/amd64 mcr.microsoft.com/devcontainers/miniconda:latest

# Update conda
RUN conda update --yes -n base -c defaults conda

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then /opt/conda/bin/conda env create -f /tmp/conda-tmp/environment.yml; fi \
    && rm -rf /tmp/conda-tmp \
    && chown -R vscode /opt/conda/envs/qfit

# Install additional OS packages for extension building.
RUN \
    apt update -y && \
    export DEBIAN_FRONTEND=noninteractive && \
    apt install -y --no-install-recommends gcc libc6-dev

# Install conda hooks for vscode user
RUN sudo -H -u vscode bash -c '/opt/conda/bin/conda init bash'

ENTRYPOINT ["/bin/bash"]
