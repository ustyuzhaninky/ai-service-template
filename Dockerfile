FROM tensorflow/tensorflow:2.4.0

ENV MODEL_DIR=./
ENV MODEL_FILE=/model.tf

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt 

COPY train.py ./train.py
COPY app.py ./app.py

RUN python3 train.py 20

COPY train.txt ./train.txt
COPY test.txt ./test.txt

EXPOSE 5000

CMD ["python3", "app.py"]