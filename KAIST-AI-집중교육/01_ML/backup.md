# Machine Learning 개론  
  
* 인공지능 개발자모임 : http://aidev.co.kr/  
 > 기본개념 학습, 세미나/포럼 등 활용가치 높음  
  
- KAIST AI집중 교육일정  
1)머신러닝   
2)딥러닝  
3)RNN   
4)CNN  
 
-실습  
act1. California Housing Price, Iris, Titanic
act2. 텐서플로우+DNN -> 영화추천시스템, 이름->국적도출  
act3. RNN -> 주식흐름판단    
act4. CNN -> 손글씨 복원  
  
- Tip
1)구글 GPU(연속12시간) 사용가능..  
2) SW Tool & 학습    
  2-1) Python: https://wikidocs.net/book/1  
  2-2) Tensorflow: https://www.tensorflow.org/guide  
  2-3) Numpy: https://docs.scipy.org/doc/numpy/    
  2-4) Matplotlib: https://matplotlib.org/index.html  
  2-5) Skearn,kunear)model.Lasso: https://scikit-learn.org/stable/  

<hr />
  
- 1일차(3/15)  

### 1. 인공신경망 학습:  
> y = f ( 합Wi Xi + b)   
> w(weight), b(constants)를 최척화하는 과정  
> 은닉층(Layer)에서  input 벡터에 W(가중치) 합산  
   -> Function(LeRu: 0미만=0,0초과=1) 통과  : constants 결정?  
   -> 에러계산(미분->최적화)  

### 2. 머신러닝 과제:  
-불충분한 data, 대표성이 없는 훈련data...  
-underfitting, Overfitting  
 > Cross-validation(test, val  set  나누고, 크로스해서 테스트)  

### 3. 머신러닝의 종류  
-supervised/un~/ Reinforcemen : 지도/비지도/강화 학습  
 >지도학습: 정답을 알려주고 학습  
 >비지도: 군집화하기  
 >강화: reward(보상)을 주는방식  

3-1)지도학습 종류(feature를 이용)   
- 용어:  
> Label : instance의 참/거짓 여부(O,X)  
> Instance: 각각의 테스트 case  
> Model: 머신러닝 메소드(테크닉) > 사용할 알고리즘  
- 종류  
 - 분류(classification)  
 - 회귀(regression) 
  : 데이터를 반복적으로 관찰하면 어떤 패턴으로 회귀한다는 것을 의미합니다.  
  : http://aidev.co.kr/learning/5264  
   >예측값이 y'고 실제값이 y라면 오차인 (y' - y)^2을 더하여 비용함수(cost function)를 만듭니다.   
    이는 w와 b의 2차 함수의 U자 모양 그래프로 나타낼 수 있는데 각각을 편미분 하여 기울기를 구하면 파라미터가 어느 방향으로 이동해야 오차를 줄일 수 있는지 알 수 있습니다.
   이렇게 '데이터 -> 오차 -> 비용함수 -> 경사하강법 학습'이 머신러닝의 공통적인 프로세스입니다.  
  : https://gdyoon.tistory.com/7  
  
  > 이 회귀 분석(Linear Regression)을 통하여 그래프 상에서 선을 그렸을 때 W(가중치) 값과 미세하게 조절되는 b(바이어스)의 값을 찾아 
    가설과 실제 데이터의 차가 가장 작은 은 최소값을 찾는 것이 머신러닝에서의 선형 회귀(Linear Regression)의 학습이라고 볼 수있다.  

- 비지도학습  
 >클러스터링: K-means,  
 >Dimesionality deduction:...  

- 학습데이터 & 품질결정요소  
> pre-processing, normaliization  
> overfitting문제 -> regulazation..  
> 학습데이터의 pre-processing  
 : Visualazation(히스토그램등) -> nomalization 필요..  
=> 개선방안: test/validation하는 데이트를 나누고, 섞어가면서 진행(page28)  

### 4. 머신러닝의 과정  
4-1) 분석  
> Big Picture  
> Get data  
> Discover,Visualize the data -> Gain Insights : data를 시각적으로 분석하여 필터링할 대상 확인  

4-2) 전처리(Pre-Processing)  
> prepare data : 공백->유효값 대체, 문자열->숫자 치환 등  
> Feature scaling : 히스토그램 시각화등 -> 평균, 표준화  

4-3) 학습(Training)  
> Select and Train a Model(메소드, 알로리즘): 사용할 AI 알고리즘 선택  

4-4) 평가  
> 결과평가(성능측정,비교)  
  
### 5. Classification  
5-1) MNIST dataset (0~9손글씨 7만장 학습)  
> data: 28X28 행렬에 0/1로 글씨(검정잉크) 존재유무 표시  
> target(lable): 0~9의 정답을 표시  
> 6만장 training set, 1만장 test set   
> shuffle the traning set  
> Traing:   
 > binary classifier(두 그룹으로 나누는방식),  
 > Stochastic Gradient Descent(SGD) classifier(SVM) : 주어진 data가 어느 카테고리에 속할지 판단  
  :https://ko.wikipedia.org/wiki/%EC%84%9C%ED%8F%AC%ED%8A%B8_%EB%B2%A1%ED%84%B0_%EB%A8%B8%EC%8B%A0  
 > Detect image  

5-2) confusion matrix > 추가정리 필요  
>precesion :   
>recall :   

5-3) Multiclass Classification (3그룹이상으로 나누는 방법)  
>...정리필요  
> Bianary Classifier : Logistic regression classifier  
> Multiclass Classifier: SoftMax classifier  


### 6. Training Model  
6-1) Linear Regression  
 > 선을 그었을때 오류가 최소가 되는 비용함수를 추론하는 최소제곱 공식...?  
 > LinearRegression(), Page64  
 > 활용사례 : 거리별 배달시간 예측 (http://woowabros.github.io/study/2018/08/01/linear_regression_qr.html)  
 > 포탈사이트 방문자 접속추세 예측: uLH 정보로 직접해보기!  


6-2) Gradient Descent Method (경사하강법)  
 > 함수의 기울기(경사)를 구하여 기울기가 낮은 쪽으로 계속 이동시켜서 극값에 이를 때까지 반복시키는 것  
 >https://ko.wikipedia.org/wiki/%EA%B2%BD%EC%82%AC_%ED%95%98%EA%B0%95%EB%B2%95  


- 종류: Batch, Mini-Batch, Stochastic  
> Batch GD :  Training Instance 전체를 테스트  
> Mini-Batch GD : Training Instance 일부씩 테스트  
> Stochastic GD :  하나의 Instance씩 테스트  
- 러닝커브  
 > 러닝코드 곡선을 시각적으로 확인하여, 오버피딩 문제를 짐작해볼수 있음  
 > 오버피팅: Training data에는 좋은성능을 가지고, Validation data에는 나쁜성능을 가지는경우 ->시각화로 예측가능  
-용어  
> hyper parameter: pre-processing에서 입력페라미터를 임으로 정하는것  
> n_epoch: 학습data를 반복 사용하는 횟수  


6-3) regularized(규칙화하는) Linear Model  
> Ridge regression : 제곱근 이용  
> Lasso regression :  |세타| 이용  
> Early Stopping : 특정반복이후 학습곡선이 나빠지면, 일찍 종료해서 성능이 좋을수있다  
  
6-4) Binary Classifier  
> Training Set Given by [ {Xi, Yi}; i=1,2..m ]  
> Given a feature vector x, evaluate p(y|x)  
  y = 1, if(p|y)>= 0.5  
         0, otherwise  

6-5) Logistic regression  
> 입력 Label이 최대화되는  y값을 추정  
  ex) feature x : (1, Age, Size) => y(양성여부: 가장 큰 가능성)를 찾는것  
> 로지스틱 모형 식은 독립 변수가 [-∞,∞]의 어느 숫자이든 상관 없이 종속 변수 또는 결과 값이 항상 범위 [0,1] 사이에 있도록 한다  
 >  로지스틱 회귀는 선형 회귀 분석과는 다르게 종속 변수가 범주형 데이터를 대상으로 하며 입력 데이터가 주어졌을 때 해당 데이터의 결과가 특정 분류로 나뉘기 때문에 일종의 분류 (classification) 기법으로도 볼 수 있다.  
 > 활용: 나이/암 사이즈별 양성여부   
 > 모델: sklearn 라이브러리 -> linear_model.LogisticRegression()  

6-6) Softmax(multiclass logistic) regression  
 >  1. X라는 데이터가 들어온다. 2. X를 A, B, C 3개의 Classifier를 통해 어디에 분류되는지 알아낸다.  
 > X가 A classifier를 통해 분류될 때,  
  1. A classifier에 의해 세워진 회귀식 (대략 Y = WX + b)에 대입한다.  
  2. 위의 Y를 Sigmoid function(S곡선 수렴..함수)으로 적용하면 0~1사이의 값(Z)이 나온다.  
  3. Z값이 1에 가까울 수록, A에 속할 확률이 높은 것으로 해석할 수 있다.  
 > Softmax알고리즘은 다음과 같은 식을 통해,  
  :위에서 나온 A, B, C에 대한 Z값들의 합이 1이 되도록 조정하는 것이다.  
  :Softmax(a) = Za / (Za + Zb + Zc)  
  :https://m.blog.naver.com/cattree_studio/220699764460  
 
> 추론과정..  
  :https://pythonkim.tistory.com/20?category=573319  
> 라이브러리(sklearn)  
 : linear_model.LogisticRegression(multi_calss="multinomial", slover="lbfgs", C=10)  


### 7. Support Vector Machine(SVM) > 정리안됨(skip)  
7-1) 만들어진 분류 모델은 데이터가 사상된 공간에서 경계로 표현되는데 SVM 알고리즘은 그 중 가장 큰 폭을 가진 경계를 찾는 알고리즘이다  
 >https://ratsgo.github.io/machine%20learning/2017/05/23/SVM/  
 >http://jaejunyoo.blogspot.com/2018/01/support-vector-machine-1.html  
7-2) SVM Kernel Trick  

<hr />

-- 실습1 (집값 예측)  
0) 파일다운로드: https://github.com/inininini/seongnam   
1) 프로그램 실행:  압축푼 파일탐색기에서 > jupyter notebook     
 > 주피터 : http://localhost:8889/notebooks/California%20Housing%20Price/Project1.%20California%20Housing%20Price.ipynb  
2) 참고: 63, 72, 76 page  
3) 실습  
 3-1) data 분석 : 문자열, 공백값등 확인  
 3-2) pre-processing: 문자->숫자치환, 공백 특정값 지정  
 3-3) 모델작성 및 학습:  모델작성?(함수  
 3-4) 학습평가:   
     
-- 실습2 (꽃잎 분류)  
1) 프로그램 실행:   
 >주피터: http://localhost:8888/notebooks/Iris/IRIS.ipynb  
2) ..  

-- 실습3(타이타닉)  
1) data분석:  
 - survived: Label(target값)  
 - 기타 변수~~:  성별등등 (peatcher:특징)  
2) pre-processig  
3) test, validation data분리 -> 학습(로딩,fit,predict)  
4) 성능평가, 이미지화  

<hr />

* 기타: 파이썬을 AI에서 주로 쓰는 이유 : http://www.ciokorea.com/news/38148  
1. 프로그래밍이 단순해진다  
2. 머신러닝 라이브러리가 있다 :  사이킷-런(Scikit-learn), 텐서플로우(TensorFlow), CNTK, 아파치 스파크 MLlib(Apache Spark MLlib),,파이토치(PyTorch) ,  
3. 메모리를 대신 관리한다  
4. 파이썬이 느려도 상관없다   
