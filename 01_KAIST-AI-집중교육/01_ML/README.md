# Machine Learning 개론 (By Scikit-learn)
  
(정리되지않은 노트)https://www.evernote.com/shard/s222/sh/8c3926e3-f8da-487f-9448-f092ca2f213a/ec63fe83701a81cdbf0ca6339283421d  
<br>

### 1. ML 기초(정의, 모델, 히스토리, Tools)  
![image](https://user-images.githubusercontent.com/45334819/61990761-a363b080-b081-11e9-8449-fc4c43e29191.png)  
- 인공신경망: 사람의 신경구조(뉴런)를 모방하여 모델링한 학습 알고리즘  
- 입력층의 신호의 총합(시그마 Wx+b)을 F(Activation Function)라는 활성화 함수를 통해 출력층으로 신호를 발산  
![image](https://user-images.githubusercontent.com/45334819/61990853-06097c00-b083-11e9-917b-a74ae8829b8d.png)  
- AI > Machine Learning > Representation Learning > Deep Learning
- Representation Learning : https://steemit.com/representation-learning/@hugmanskj/representation-learning  
<br>

### 2. Types of Machine Learning Systems (Supervised, Unsupervised, Reinforcement)   
![image](https://user-images.githubusercontent.com/45334819/61990966-db202780-b084-11e9-98ff-e36130d6a8eb.png)
<br>

### 3. End-to-End Machine Learning (머신러닝의 과정)
#### 1)분석
- Big Picture
- Get data
- Discover,Visualize the data -> <b>Gain Insights</b> : data를 시각적으로 분석하여 필터링할 대상 확인  
#### 2)전처리(Pre-Processing)
- prepare data : 공백->유효값 대체, 문자열->숫자 치환 등
- Feature scaling : 히스토그램 시각화등 -> 평균, 표준화
#### 3)학습(Training)
- Select and Train a Model(메소드, 알로리즘): 사용할 AI 알고리즘 선택
#### 4)평가
- 결과평가(성능측정,비교)
<br>

### 4. Classification (MNIST dataset)    
- MNIST dataset (0~9손글씨 7만장 학습)  
> data: 28X28 행렬에 0/1로 글씨(검정잉크) 존재유무 표시  
> target(lable): 0~9의 정답을 표시  
> 6만장 training set, 1만장 test set 
- Traing: 
> binary classifier(두 그룹으로 나누는방식)  
> multinomial classifier(3그룹이상으로 나누는 방식, ex: Softmax Logistic Regression)  
<br>

### 5. Performance Measures (Confusion Matrix, Precision & Recall, ROC curve)  
![image](https://user-images.githubusercontent.com/45334819/61991355-c5613100-b089-11e9-9bc7-76042743f9e5.png)  
![image](https://user-images.githubusercontent.com/45334819/61991357-cb571200-b089-11e9-9a51-ab7949bab4a3.png)  
- 참고:  https://www.waytoliah.com/1222  
<br>

### 6. Multiclass Classification
<br>

### 7. Training Models (Linear Regression)
<br>

### 8. Gradient Descent Method
<br>

### 9. Regularized Linear Model
<br>

### 10. Logistic regression
<br>

### 11. Softmax(Multicalss logistic) regression
<br>

### 12. SVN
<br>

<hr>

### 실습  
