# 03_Tensorflow_알고리즘구현

- 참고 사이트 : https://hunkim.github.io/ml/
<hr />

### Sumarry  

## 1. Tensorflow (graph, tensor, session ...)  
- Tensorflow 동작 메커니즘: Node(Tensor)를 연결하는 Graph를 그리고, 그래프에 Data 세팅(Feed)하고 수행하기 위해 Session을 구동한뒤, 결과를 그래프에 업데이트 한다.
- placeholder : 노드(Tensor)에 담을 공간을 placeholder로 마련한뒤, 추후 session running시에 값을 feed하여 처리
- Rank: 차원(1~N차원 array), <b> Shapes: 해당 array의 변수갯수 </b>
- 소스예제 : https://github.com/jukyellow/artificial-intelligence-study/blob/master/03_Tensorflow_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EA%B5%AC%ED%98%84/01_basic_tensorflow.py   
![image](https://user-images.githubusercontent.com/45334819/58370209-d7fea300-7f3e-11e9-9f1a-1b8e9eed8b00.png)   
<hr />

## 2. Thery (Hypersis, Cost, Weight, Bias ...)  
- 가설(Hypersis)과 실제 결과값의 차이(Cost or Loss)를 최소화 하도록 Weight(모델의 기울기값)와 Bias(초기값)를 반복적인 학습을 통해서 조정함.
- 소스예제 : https://github.com/jukyellow/artificial-intelligence-study/blob/master/03_Tensorflow_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EA%B5%AC%ED%98%84/03_linear_regression_tensorflow.ipynb  
![image](https://user-images.githubusercontent.com/45334819/58370210-dd5bed80-7f3e-11e9-8038-e20ae31d3005.png)  
<hr />

## 3. linear regression  
- 경사하강법(Gradient Descent) 유도(Derivation) : Cost함수의 순간 기울기(미분) 구하여 Plus나 Minus방향으로 Cost가 최소화가 되도록 이동  
- Tensorflow Minimize Gradient Descent Optimizer   
> ex:  
> train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)  
- 소스예제 : https://github.com/jukyellow/artificial-intelligence-study/blob/master/03_Tensorflow_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EA%B5%AC%ED%98%84/04_3_linear_regression_tf_GradientDescentOptimizer.ipynb  
![image](https://user-images.githubusercontent.com/45334819/58370213-e056de00-7f3e-11e9-9528-6ff0f80bb500.png)  
<hr />

## 4. Matrix(행렬) 연산  
- Matrix multiplication
![image](https://user-images.githubusercontent.com/45334819/58432740-decc1800-80ee-11e9-95a4-40d7c168ce50.png)  
- numpy slice
![image](https://user-images.githubusercontent.com/45334819/58432742-e2f83580-80ee-11e9-8c0c-3d33243386a2.png)  
![image](https://user-images.githubusercontent.com/45334819/59210253-2622d000-8be8-11e9-8bf6-bb04545303ba.png)  
- google colab fileUpload   
<pre>
#파일업로드창 출력  
uploaded = files.upload()  
  
#업로드한 파일정보 출력  
for fn in uploaded.keys():  
 print('Upload file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))  
</pre>  

## 5. Logistic Regression  
- 선형함수(wX+b)에서 예외적인 학습data(튀는값)이 발생할때 오차가 커지는 문제를 개선하기위해,  
  Activation(활성화) 함수로 sigmoid함수를 사용해서 0~1로 수렴하는 H(x)가설(hypothesis)를 찾아낸다.  
- 이때, Cost함수는 local-minimum이 발생할수 있는데, -Log를 취해서 global-minimum을 찾아갈수 있도록 했다.  
![image](https://user-images.githubusercontent.com/45334819/58574443-f1735800-827a-11e9-9e1b-6a9837355a7a.png) 
- tensorflow 소스에서는 수식그대로 구현하면 됨.  
<pre>
# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))  
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)  
# cost/loss function  
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))  
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)  
</pre>
- sckit-learn libarary 예제: https://github.com/jukyellow/artificial-intelligence-study/blob/master/03_Tensorflow_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EA%B5%AC%ED%98%84/05_2-logistic_regression_by_sckit_learn.ipynb
<hr />


## 6. Softmax Regression  
- Softmax Regression(Multicasss Logistic) : 여러개의 class중에 확률이 가장 큰 값을 선택하여 one-hot encoding으로 표시  
- one-hot encoding: 여러 class를 확률이 가장 높은 값을 1, 그외는 0으로 표시  
- cross entropy(혼잡도, 비): cost function에서 cost 또는 loss를 의미  
- reshape : one-hot encoding등에서 출력값의 array갯수를 맞추기위해 shape를 변경하는 것  
![image](https://user-images.githubusercontent.com/45334819/59210684-18ba1580-8be9-11e9-97d5-f9b3ea8ce9fc.png)
![image](https://user-images.githubusercontent.com/45334819/59210689-1c4d9c80-8be9-11e9-8085-f78ee52ca7a1.png)

- tensorflow softmax 예제: https://github.com/jukyellow/artificial-intelligence-study/blob/master/03_Tensorflow_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EA%B5%AC%ED%98%84/06_1_softmax_classifier.ipynb  
- tensorflow softmax_cross_entropy_with_logits 예제: https://github.com/jukyellow/artificial-intelligence-study/blob/master/03_Tensorflow_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EA%B5%AC%ED%98%84/06_2_softmax_cross_entropy_with_logits.ipynb  
- sckit-learn libarary 예제: https://github.com/jukyellow/artificial-intelligence-study/blob/master/03_Tensorflow_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EA%B5%AC%ED%98%84/06_3_softmax_classifier_by_sckit_learn.ipynb  
<hr />  

## 7. ML의 실용과 몇가지 팁  
### Learning-rate, Preprocessing(Normalization), Overfitting(Regularization):  
 1. Learning-rate  
  - 보통 0.01정도에서 시작  
  - cost변화를 관찰할 필요있음  
  - 너무크면, weight최소점을 찾지못하고 헤메고, 너무 작으면 빨리 수렴하지 않음 
  ![image](https://user-images.githubusercontent.com/45334819/59365066-dcb2bc00-8d72-11e9-818d-5a84cb6818d4.png)
  - 예제: https://github.com/jukyellow/artificial-intelligence-study/blob/master/03_Tensorflow_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EA%B5%AC%ED%98%84/07_1_Learing_rate_test.ipynb  
 2. Preprocessing (->Normalization: 정규화) 
  - 입력값(X)의 분포가 일정하도록 조정하여 좋은 성능을 갖도록 함(학습이 잘 안될때 확인필요)
  - ex1) Standarization(표준화):  
  ![image](https://user-images.githubusercontent.com/45334819/59364987-b856df80-8d72-11e9-9a6d-79b004c8272b.png)
  - ex2) MinMax Scaling  
  - 예제: https://github.com/jukyellow/artificial-intelligence-study/blob/master/03_Tensorflow_%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98%EA%B5%AC%ED%98%84/07_2_linear_regression_min_max(normalization).ipynb  
 3. Overfitting  
  - 학습 data에만 최적하된 모델을 만들어, 테스트 data에 성능이 좋지 않음 (Weight가 특정값에 크게 튀면서 구브러짐)
  - Regularization(일반화): 너무 큰 입력값에 weight가 구부러지지 않게 펴는 효과
    (Regularization을 위해 Weight값을 제곱하고 평균낸 값을 더함 ) 
  ![image](https://user-images.githubusercontent.com/45334819/59364991-ba20a300-8d72-11e9-8946-c6f5dbe517fa.png)


<hr />



 

