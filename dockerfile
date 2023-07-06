############################################# INSTRUCTIONS #############################################
# build image: docker build -t pyraws:latest  -f dockerfile .  
# run image: docker run -it --rm -p 8888:8888 pyraws:latest                
#############################################     END     #############################################



# Use the official PyTorch base image
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /workdir

# Install Python dependencies for pyraws:
RUN apt-get update && apt-get install -y git
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# install pyraws
RUN git clone https://github.com/ESA-PhiLab/PyRawS.git
RUN cd PyRawS && source pyraws_install.sh


