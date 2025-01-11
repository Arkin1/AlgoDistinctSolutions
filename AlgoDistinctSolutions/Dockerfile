FROM tensorflow/tensorflow:latest-gpu

RUN apt update -y && apt install cppcheck git -y

WORKDIR /app
RUN git clone https://github.com/dspinellis/tokenizer.git && make -C /app/tokenizer/src