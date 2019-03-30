# RNN(Recurrent Neural Network, 순환신경망)


1. RNN개념 (시계열, 연결, 메모리, 순환구조, 신경망)
> 순서가 존재하는 시계열 DATA(길이가 일정 or 길이가 일정하지 않은)를 순환구조의 graph로 연결하여  
 이전 연산의 결과를 메모리에 저장하고 새로운 연산의 입력으로 반복해 사용하는 인공신경망 모델이다.  
 ![image](https://user-images.githubusercontent.com/45334819/55249812-4e10c180-5290-11e9-845b-b7c0c178276b.png)

 
- SRN(Simple Recurrent Neural Network) 기본함수  
![SRN](https://user-images.githubusercontent.com/45334819/55276744-8076f900-533a-11e9-8b51-e78772b4ecbf.jpg)  
>함수: h(t) = tanh( Wh(t-1) + Ux(t) + b)  
>설명: 시간 t에 따른 Hidden Layer h(t)는, Hidden Layer(t-1)과 W(weight)의 합성 + U(특정사상행렬)와 x(i)입력값의 합성 + bias(상수값)를  활성화에 적용   
>Prameter set : W, U, b  
>부가설명:  시계열적 data를 선형적으로 선언하여 비선형함수를 통과시켜 시간에 따른 h변화 함수를 구한다  
 
- RNN활용분야  
1)Image Captioning: 그림->text 추출  
2)Activity Classification : 영상-> 액션분류  
3)Machine Translation: 번역기  
4)Speech recognition: 음성인식  


2. RNN 학습, BPTT(BackPropagation through time)  
![RNN](https://user-images.githubusercontent.com/45334819/55276773-c6cc5800-533a-11e9-8adf-bad243f03523.jpg)  
- 예제: http://solarisailab.com/archives/1451  
- RNN학습  
 1)cross-entropy loss L(손실함수) = 시그마( L(Y,Y^)): 매 timestamp에서 예측값과 실제값의 차이의 합 = -시그마(t)시그마(i)(Yi(t) logY^i(t)  
 2)Y^(t) = softmax(Vh(t) + c)  
 3)h(t)  = tanh( Wh(t-1) + Ux(t) + b)  

- BPTT  
![BPTT](https://user-images.githubusercontent.com/45334819/55276776-da77be80-533a-11e9-9314-b8a9bffed12b.jpg)  
> http://solarisailab.com/archives/1451


-RNN 간단한 실습예제(hello)
>https://mobicon.tistory.com/537
  
  
- RNN 학습 어려움과 개선방안
1)Overfitting -> Early Stopping, Dropout, L1/L2 regularization(특정 페라미터를 추가해서 학습을 어렵게 만듦)   
2)Exploding gradient -> Gradient clipping  
3)Vanishing gradient -> LSTM, GRU 모델 사용  
>개선방안 예제:  >https://ratsgo.github.io/deep%20learning/2017/10/10/RNNsty/  
  
    
3. RNN 활용영역  
1)Time series regression: 시간대별 승객수 추이예측  
2)Language modeling: Recurrent neural nw lan model (RNN처럼 이전문장단어를 기억하고 다음연산에 활용)  

- RNN의 변형  
1)Bidirectional RNN  
2)Encoder-Decoder(번역),   
3)Attention...  
  
- LSTM (), GRU()  
![LSTM](https://user-images.githubusercontent.com/45334819/55276778-e6fc1700-533a-11e9-9808-115c54c19c98.jpg)  
  
![GRU](https://user-images.githubusercontent.com/45334819/55276781-f4190600-533a-11e9-9ab5-ea25d83ca9e3.jpg)    
  
  
- 기타 용어:  
`hyperbolic(쌍곡선) tangent(tanh) : -1 ~ 1 사이의 비선형적 값을 갖는 활성화 함수  
`엔트로피(entropy):   흔히 일반인들에게 무질서도라고 알려져있기도 하지만 정확한 개념의 이해를 위해 좋은 말은 아니다.[1] 다른 말로 설명하자면 물질의 상태 변화, 즉 에너지의 전환 과정에서 유용하게 쓸 수 있는 에너지가 줄어드는 정도를 의미한다  
`gradient: 기울기  
`vanishing: 사라지는  
`exploded: 폭발하는  
`활성화함수(activation function): 입력값의 총합을 출력신호로 변환하는 함수(0 or 1로 수렴). ex) 시그모이드,계단함수, ReLU, tanh  
  

4. RNN 학습로직 구현방법  
1)Placeholders 세팅: feed data into graph  
2)Model selection: make prediction of given data  
3)Loss function : how well does the mode work   
4)Optimizer: SGD algorithm that optimizes model weight and bias.  

5. 실습
5-1) tensorflow 모델(RNN, LSTM, GRU)로 주식data 학습모델 최적화(python, Jupyter)  
5-2) RNN, LSTM, GRU python   
