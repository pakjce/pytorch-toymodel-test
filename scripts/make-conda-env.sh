#!/usr/bin/env bash
set -ef -o pipefail

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
CONDA_ENV_NAME=pytorch15-study
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

conda create -n ${CONDA_ENV_NAME} python=3.6

conda install -n ${CONDA_ENV_NAME} -y uwsgi -c conda-forge
conda install -n ${CONDA_ENV_NAME} -y virtualenv numba cython numpy \
    mkl mkl-include setuptools jupyterlab jupyter \
    matplotlib protobuf pytorch=1.5 torchvision cpuonly wheel -c pytorch

# Install Essential Packages
CONDA_ENV_DIR=$(conda env list | grep "/.*/${CONDA_ENV_NAME}" | awk '{print $2}')

if [[ -z "$CONDA_ENV_DIR" ]]
then
      echo "Fail to find conda env!"
      exit
else
      ${CONDA_ENV_DIR}/bin/pip install awscli dvc[s3] pipenv
fi
