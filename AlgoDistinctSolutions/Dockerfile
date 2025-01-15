FROM tensorflow/tensorflow:latest-gpu

# RUN apt update && apt upgrade -y && apt install wget git -y

# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
# ENV PATH="/root/miniconda/bin:${PATH}"

# RUN conda create -y -n ml_env python=3.10

# RUN conda init && . /root/.bashrc && conda activate ml_env && pip install tensorflow[and-cuda]==2.10.0

# RUN apt update -y && apt install cppcheck git -y

# WORKDIR /app

# RUN git clone https://github.com/dspinellis/tokenizer.git && make -C /app/tokenizer/src
# RUN git clone https://github.com/radareorg/radare2 && radare2/sys/install.sh

# RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# ARG ProjectName=AlgoDistinctSolutions

# COPY requirements.txt $ProjectName/requirements.txt
# RUN pip install --no-cache-dir -r $ProjectName/requirements.txt 

# COPY dependencies  $ProjectName/dependencies
# RUN pip install --no-cache-dir ./$ProjectName/dependencies/safe

# COPY . $ProjectName