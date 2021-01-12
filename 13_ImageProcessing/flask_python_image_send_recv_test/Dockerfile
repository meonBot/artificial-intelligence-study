FROM jukyellow/keras-flask:cpu
#FROM flask/deepo
#MAINTAINER juk "jukyellow@ktnet.co.kr"

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential

COPY . /app
WORKDIR /app

#RUN pip install -r requirements.txt

ENTRYPOINT ["python"]
CMD ["FaceNet_Server.py"]
#CMD ["goodsdesc_keras_flask_api_server.py"]