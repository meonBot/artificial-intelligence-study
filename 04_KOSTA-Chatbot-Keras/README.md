# 04_KOSTA-Chabot(Word Embedding)-By-Keras

- note: https://www.evernote.com/shard/s222/sh/66b41ec4-10a6-41e7-80e8-1d0648008f98/4869a90fd0f812a06d9de4e3d15129a7

<hr />
<br>

## Word Embedding 정리  
<br>

### 1. 개념정리

#### 1-1. Bag of Words  
- 딥러닝 이전에 가장 많이 사용되던 모델   
- 문장에서 나타난 단어들의 인덱스를 저장(보통 빈도수도 같이 저장)  
- 보통 One-Hot 인코딩을 사용(굉장히 코그 sparse한 벡터, 영어의 경우 2만(20k)의 단어가 사용??)  
  1. 단어들의 관계가 전혀 포현되지 못함  
  2. 단어의 순서도 표현이 안됨  
> 옥스포드 영어사전은 단어가 75만개라고 함  
- 보통 3개의 특수단어 존재: SOS(Start of Sentence), EOS(End Of Sentense), UNKNOWN(모르는 단어)  
![image](https://user-images.githubusercontent.com/45334819/61086181-245d5e00-a46d-11e9-8b82-02ab55331f1a.png)

#### 1-2. Word Embedding
##### word embedding 이란?
- W/E 목표: 벡터 공간의 축소(Dense), 단어간의 의미관계가 벤턱 공간에 기하 관계로 표현  
![image](https://user-images.githubusercontent.com/45334819/61086658-5d4a0280-a46e-11e9-8b30-9a578f847b90.png)

- 두가지 방식의 모델 트레이닝
 1. CBOW(Continuos bag of words) : 중심단어를 비우고 window를 옮기면서 중심단어를 예측하는 방식
 2. Skip-gram : 중심단어 왼쪽/오른쪽편 단어를 예측하는 방식  
*Skip-gram 상세 정리 필요  

##### Pre-Trained Word Embedding
- word2vec(google) : 2013,
- fastText(Facebook) : 2015, skip-gram(n:3 or 6)을 활용해 새로운 단어가 나타나도 단어의 조합으로 분류가능
- glove(스탠포드) : 2016, 사전에 나타난 단어의 확률에 바탕

##### W/E Layer 이용한 단어변환
- 

##### W/E 사용밥법
* 충분한 data가 있는 경우 : CBOW, Skip-Gram을 이용해 Embedding 모델을 생성
* 데이터가 충분히 있지 않거나, 범용 모델로 충분하다면 Pre-Trained된 모델을 다운받아 사용

<hr />
<br>


## Language Model
<br>

### 개요
- 정의 : 다음에 나오는 단어가 무엇인지 예측하는 것
> The Students opened their ________
> 예측ex: books, laptops, exams ...
- 최근 가장 유명한 Language Model이 BERT

### 다음 단어 예측하기(예제)


