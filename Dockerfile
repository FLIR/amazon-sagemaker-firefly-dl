# The offica Nvidia docker package
FROM asigiuk/tf1.13-ncsdk-gpu-runtime:latest


# Install sagemaker-training toolkit that contains the common functionality necessary to create a container compatible with SageMaker and the Python SDK.
RUN pip install sagemaker-training

# Copies the training code inside the container
RUN mkdir /opt/ml
RUN cd /opt/ml && git clone -b aws_sagemaker https://github.com/FLIR/iis_firefly_image_classifier.git
RUN cp -a /opt/ml/iis_firefly_image_classifier /opt/ml/code
RUN ls /opt/ml/code/imagenet_checkpoints/

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM train_image_classifier.py
#ENV SAGEMAKER_PROGRAM test.py
