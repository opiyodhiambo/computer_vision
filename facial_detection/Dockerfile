FROM python:3
RUN pip install --upgrade -r /app/requirements.txt
RUN mkdir -p /app
COPY ./model/ /app/
COPY ./datasetloader.py /app/datasetloader.py
COPY ./utils.py /app/utils.py
COPY ./requirements.txt /app/requirements.txt 
COPY ./facial-detect.py /app/facial-detect.py
WORKDIR /app
CMD ["python", "facial-detect.py"]
