#!/usr/bin/env bash
set -e -o pipefail

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
CONDA_ENV_NAME=pytorch15-study
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

PYTHON_ENV_DIR=$(conda env list | grep "/.*/${CONDA_ENV_NAME}" | awk '{print $2}')

if [[ -z "$PYTHON_ENV_DIR" ]]
then
      echo "Fail to find conda env!"
      exit
fi


PYTHON_ROOT=${PYTHON_ENV_DIR}
PYTHON_EXEC=${PYTHON_ROOT}/bin/python3
PIPENV_EXEC=${PYTHON_ROOT}/bin/pipenv

if [[ ! -d "${PYTHON_ROOT}/bin"  ]]
then
    echo "${PYTHON_ROOT}/bin 폴더가 존재하지 않습니다!"
    exit 1
fi

if [[ ! -e "${PYTHON_EXEC}" ]]
then
    echo "${PYTHON_EXEC} 가 존재하지 않습니다!"
    exit 1
fi

if [[ ! -e "${PIPENV_EXEC}" ]]
then
    echo "${PIPENV_EXEC} 가 존재하지 않습니다!"
    exit 1
fi

echo "+ PYTHON DIR: ${PYTHON_ROOT}"
echo "+ PYTHON EXEC: ${PYTHON_EXEC}"
echo "+ PIPENV EXEC: ${PIPENV_EXEC}"

rm -rf .venv

export PIPENV_VENV_IN_PROJECT=true
export PIPENV_IGNORE_VIRTUALENVS=1
${PIPENV_EXEC} --site-packages --python ${PYTHON_EXEC}

LIB_LIST=$(ls ${PYTHON_ROOT}/lib/libpython*.*)
for libfile in $LIB_LIST
do
    FILENAME=$(basename ${libfile})
    if [[ ! -e .venv/lib/${FILENAME} ]]; then
        ln -sf ${libfile} .venv/lib/${FILENAME}
    fi
done
