# Python 기본문법

### 참고1. 밑바닥부터 시작하는 딥러닝
 - 기본문법(변수,연산,함수,클래스,라이브러리 로딩, 그래프 등)  
 https://www.evernote.com/shard/s222/sh/cbdefc88-41d7-4b88-b718-aaf9659262c0/15f336ad07be694497a3c586a0ed14cc 
 
 - squeeze : 차원축소  
 https://www.evernote.com/shard/s222/sh/43636d07-ca5b-4a18-93e9-6574e5032046/93b5d84a57ceb3258516c410a58cd0e1  
 
 - argmax/axis : 차원별 최대값의 index  
 https://www.evernote.com/shard/s222/sh/11ed9209-4b4f-4c89-a083-05ac94f1a45a/202d75ccc343ca82fb96ef1b6c2dd97f  
 
 - key/value 집합(dictional) 소팅
 https://www.evernote.com/shard/s222/sh/a6aa8211-6e2b-4b7b-a69f-9e85e63381c4/4b9755151f266156034026e077d7e314  
 
 
 

<hr />

### 참고2. 웹사이트
 - 점푸투파이썬: https://wikidocs.net/13  
 - 파이썬 자습서: https://docs.python.org/ko/3/tutorial/index.html  
 - 왕초보를 위한 파이썬: https://wikidocs.net/77  
#### 1. 

<hr />


#### 1. pickle을 이용한 dictionary 저장/로딩
```
import pickle

dic = {}
dic["word"] = ['a', 'b', 'c']
dic["weight"] = ['1.1', '2.5', '0.1']
dic["hs"] = ['1234', '5678', '0001']

print(dic)
>> {'word': ['a', 'b', 'c'], 'weight': ['1.1', '2.5', '0.1'], 'hs': ['1234', '5678', '0001']}

file_name='dic_test.pickle'
with open(file_name, 'wb') as w_handler:
    pickle.dump(dic, w_handler, protocol=pickle.HIGHEST_PROTOCOL)

with open(file_name, 'rb') as r_handler:
    l_dic = pickle.load(r_handler)

print('loaded_dic:', l_dic)      
>> loaded_dic: {'word': ['a', 'b', 'c'], 'weight': ['1.1', '2.5', '0.1'], 'hs': ['1234', '5678', '0001']}
```

#### 5. python list정렬 후 정렬전 index 뽑기
: 05_python_dictionary_sort.py
