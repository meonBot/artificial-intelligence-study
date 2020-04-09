# Google AutoML

- https://console.cloud.google.com/natural-language/dashboard

- 제약사항  
1. 최대 100만건까지 로딩가능  
2. data 중복(동일 data + 동일 label)시 오류발생(중복제거후 로딩해야함)  
3. text와 label은 콤마로 분리(data에 콤마 포함시 오류발생함)  

- 장점
1. 평가 결과를 시각화해 줌(혼돈모델 등)  


![automl_1](https://user-images.githubusercontent.com/45334819/77118863-8c425a80-6a78-11ea-9ea5-152725a2521d.jpg)

![automl_2](https://user-images.githubusercontent.com/45334819/77118866-8d738780-6a78-11ea-95ca-9884d726c136.jpg)

![image](https://user-images.githubusercontent.com/45334819/77119171-44700300-6a79-11ea-9808-3b3ba80e8777.png)  

- 단점
1. 비용: 2~3일 학습용으로 사용했는데 50$ 청구됨  
