FROM  tensorflow/tensorflow:2.17.0-gpu
WORKDIR /app

RUN apt update && apt upgrade -y && apt install wget git cppcheck build-essential -y 

RUN git clone https://github.com/dspinellis/tokenizer.git && make -C /app/tokenizer/src
RUN git clone https://github.com/radareorg/radare2 && radare2/sys/install.sh


RUN wget https://bootstrap.pypa.io/get-pip.py && python get-pip.py
RUN pip install --no-cache-dir torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118

# ARG ProjectName=AlgoDistinctSolutions

# # COPY requirements.txt $ProjectName/requirements.txt
# # RUN pip install --no-cache-dir -r $ProjectName/requirements.txt 

# COPY . $ProjectName