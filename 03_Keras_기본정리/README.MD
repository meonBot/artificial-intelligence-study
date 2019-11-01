# Keras 기본문법 및 알고리즘 구현 정리  


### 1. Input Layer, Model 클래스  



### 2. attention관련 keras class/function 정리  

1. Dense : 입력과 출력을 모두 연결(전결합층, FullyConnectedLayer)  
2. Permute : transpose와 유사한 역할  
3. Lambda Layer: 임의 표현을 Layer 객체로 래핑  
 - lambda식 : 식별자 없이 실행 가능한 함수 표현식(익명함수)  
4. backend.mean: 입력 tensor의 평균값  
5. RepeatVector : Repeats the input n times.  
- model.add(RepeatVector(3))  
- ex: model.output_shape == (None, 3, 32)  
6. tensorflow merge = keras Multifly() --concatenate(merge)  
- Merge all available features into a single large vector via concatenation  
- ex) attention_mul = multiply([inputs, attention_probs], name='attention_mul')  
7. Flatten : 입력vector를 일차원으로 바꿈  
  
  
