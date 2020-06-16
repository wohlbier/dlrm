# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

ARG FROM_IMAGE_NAME=pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
FROM ${FROM_IMAGE_NAME}

ARG PROXY
ENV http_proxy $PROXY
ENV https_proxy $PROXY

ADD requirements.txt .
RUN pip install -r requirements.txt
WORKDIR /code/train
ADD . .

ENTRYPOINT ["/bin/bash"]
