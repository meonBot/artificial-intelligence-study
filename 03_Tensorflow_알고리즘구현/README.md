# 03_Tensorflow_알고리즘구현

- 참고 사이트 : https://hunkim.github.io/ml/
<hr />

### Sumarry  

### 1. Tensorflow (graph, tensor, session ...)  
![image](https://user-images.githubusercontent.com/45334819/58370209-d7fea300-7f3e-11e9-9f1a-1b8e9eed8b00.png)  
- Tensorflow 동작 메커니즘: Node(Tensor)를 연결하는 Graph를 그리고, 그래프에 Data 세팅(Feed)하고 수행하기 위해 Session을 구동한뒤, 결과를 그래프에 업데이트 한다.
- 소스예제 : https://github.com/jukyellow/artificial-intelligence-study/blob/master/03_Tensorflow_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EA%B5%AC%ED%98%84/02_basic_tensorflow.py  
<hr />

### 2. Thery (Hypersis, Cost, Weight, Bias ...)  
![image](https://user-images.githubusercontent.com/45334819/58370210-dd5bed80-7f3e-11e9-8038-e20ae31d3005.png)  
- 가설(Hypersis)과 실제 결과값의 차이(Cost or Loss)를 최소화 하도록 Weight(모델의 기울기값)와 Bias(초기값)를 반복적인 학습을 통해서 조정함.
- 소스예제 : https://github.com/jukyellow/artificial-intelligence-study/blob/master/03_Tensorflow_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EA%B5%AC%ED%98%84/03_linear_regression_tensorflow.ipynb  
<hr />

### 3. linear regression  
![image](https://user-images.githubusercontent.com/45334819/58370213-e056de00-7f3e-11e9-9528-6ff0f80bb500.png)  
- 경사하강법(Gradient Descent) 유도(Derivation) : Cost함수의 순간 기울기(미분) 구하여 Plus나 Minus방향으로 Cost가 최소화가 되도록 이동  
- Tensorflow Minimize Gradient Descent Optimizer   
> ex:  
> train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)  
- 소스예제 : https://github.com/jukyellow/artificial-intelligence-study/blob/master/03_Tensorflow_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EA%B5%AC%ED%98%84/04_3_linear_regression_tf_GradientDescentOptimizer.ipynb  
<hr />

### 4. Matrix(행렬) 연산  


### 5. Logistic Regression  


 

