#!/usr/bin/env bash

source /opt/conda/etc/profile.d/conda.sh \
&& conda activate qfit \
&& pip install -r .devcontainer/requirements.develop.txt
