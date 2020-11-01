FROM tensorflow/tensorflow:2.3.0-gpu

# Install pip3
RUN apt-get update && \
    apt-get install -y python3-pip 

RUN python3 -m pip install --upgrade pip
# Install spyder
#RUN pip3 install spyder

RUN pip3 install scikit-image


RUN pip3 install jupyterlab

WORKDIR /ct_test
CMD jupyter-lab  --allow-root  --no-browser --ip=0.0.0.0 # --notebook-dir=/ct_test # ip=127.0.0.1 does not work


#sudo docker build -t sang/ct_test .
#sudo docker run --gpus all -it -p 8888:8888 -v /media/data/sang_data/ct_test:/ct_test   sang/ct_test

