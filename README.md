# pytorch-toymodel-test

Pytorch 1.5 Toymodel을 Test하기 위한 환경입니다.

## 1. 환경 구성 방법

**중요: Conda가 설치되어있어야 합니다.**

1) `./scripts/make-conda-env.sh` 를 실행하여 pytorch 1.5 CPU 버전이 포함된 conda 환경을 생성합니다.

2) `./scripts/make-virtualenv.sh` 를 실행하여 1) 에서 생성한 conda 환경 기반으로 Pipenv 기반 가상환경을 project 폴더에 생성합니다.

## 2. 환경 activate 방법

1) `source ./scripts/activate-conda-env.sh` 를 실행하여 현재 Shell 환경을 pytorch 1.5 CPU 버전 기반 conda 환경으로 전환합니다.

2) `pipenv shell` 로 현재 project 기반 virtualenv 로 전환합니다.

### 환경 구성 Tip

[direnv](https://direnv.net/) 를 설치해두면 다음과 같이 project 폴더에 `.envrc` 파일을 만들어서 자동으로 환경이 activate 되도록 할 수 있습니다.

```bash
source ./scripts/activate-conda-env.sh
```
