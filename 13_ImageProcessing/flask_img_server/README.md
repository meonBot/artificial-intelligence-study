# flask image server 설치/구동

### 1. 도커 img 다운로드
> keras-flask-img 다운로드 : https://hub.docker.com/repository/docker/jukyellow/keras-flask-img
```
docker pull jukyellow/keras-flask-img:cpu-sklearn
```

### 2. 도커 face-net 서버 설치/구동
```
docker build -t facenet-server .
docker run --name facenet-server --publish 8312:8312 -it facenet-server
```

