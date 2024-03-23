############################################# INSTRUCTIONS #############################################
# build image: docker build -t pyraws:latest  --build-arg CACHEBUST=$(date +%s) -f dockerfile . 
# run image: docker run -it --rm -p 8888:8888 pyraws:latest                
#############################################     END     #############################################

# Use the official PyTorch base image
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /workspace

# Install Python dependencies for pyraws:
RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6
# install pyraws

RUN git clone -b main https://github.com/ESA-PhiLab/PyRawS.git

RUN cd PyRawS && /bin/bash -c "source ./pyraws_install.sh" 
RUN echo "source activate pyraws" > ~/.bashrc
ENV PATH /opt/conda/envs/pyraws/bin:$PATH
