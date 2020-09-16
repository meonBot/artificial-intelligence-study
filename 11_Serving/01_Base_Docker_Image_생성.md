
# base 이미지 직접 생성하기

### 1.필수 패키지(keras serving)  
```
tensorflow  
keras
sklearn
flask
```

### 2. 도커 허브 다운로드
> docker pull tensorflow/tensorflow

### 3. tensorflow 버전(latest) 확인하기
> docker run -p 9001:9001 -it tensorflow/tensorflow /bin/bash  
> pip freeze   
: tensorflow==2.3.0  

### 4. tensorflow 이미지에 추가 패키지 인스톨
> (https://m.blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221338278344&proxyReferer=https:%2F%2Fwww.google.com%2F)  
> pip install keras  
> pip install sklearn  
> pip install flask  

: keras-2.4.3 pyyaml-5.3.1  
: joblib-0.16.0 scikit-learn-0.23.2 sklearn-0.0 threadpoolctl-2.1.0  
: Jinja2-2.11.2 MarkupSafe-1.1.1 click-7.1.2 flask-1.1.2 itsdangerous-1.1.0  

### 5.이미지 commit(굽기)
> 4)번에서 install후 종료하기 전에, 다른 cmd창에서 해당 이미지 굽기  
> docker commit competent_proskuriakova tf/keras-flask  

> 1.54G(tensorflow) -> 1.58G(keras/sklearn/flask)  

### 6. docker hub 올리기  
> 계정가입  


