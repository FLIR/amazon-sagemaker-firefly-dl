# The offica Nvidia docker package
FROM nvcr.io/nvidia/cuda:10.0-cudnn7-runtime-ubuntu16.04



RUN apt-get --fix-missing update
RUN apt-get update

# Install basic apps
RUN apt-get install -y -q software-properties-common \
													build-essential \
													cmake \
													checkinstall \
													pkg-config \
  												wget git curl \
  												unzip yasm x11-apps\
													nano vim sudo

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update


#### Python 3.7
# python libraries
RUN apt-get install -y -q python3-pip python3.7-dev
RUN apt-get install -y -q  python3.7
RUN ln -s python3.7 /usr/bin/python

# Install the latest version of pip (https://pip.pypa.io/en/stable/installing/#using-linux-package-managers)
RUN wget --no-check-certificate  https://bootstrap.pypa.io/get-pip.py
RUN python get-pip.py


### Tensorflow-GPU, TensorFlow-slim, and Guildai and other useful libraries


## Specify tensorflow version

# Install extra packages without root privilege if need
RUN pip install tensorflow-gpu==1.13.2 scikit-learn

# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.
RUN pip install sagemaker-training

# Copies the training code inside the container
RUN mkdir /opt/ml
RUN cd /opt/ml && git clone -b aws_sagemaker https://github.com/FLIR/iis_firefly_image_classifier.git
RUN cp -a /opt/ml/iis_firefly_image_classifier /opt/ml/code
RUN ls /opt/ml/code/imagenet_checkpoints/mobilenet_v1_1.0_224

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train_image_classifier.py
