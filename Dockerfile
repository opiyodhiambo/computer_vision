FROM python:3
RUN mkdir -p /app
COPY ./model/ /app/
COPY ./datasetloader.py /app/datasetloader.py
COPY ./utils.py /app/utils.py
COPY ./requirements.txt /app/requirements.txt 
COPY ./facial_detect.py /app/facial_detect.py
RUN pip install --upgrade -r /app/requirements.txt
WORKDIR /app
CMD ["python", "facial_detect.py"]
