# One-Hot-Encoding 메모리 문제해결 -> sparse_categorical_crossentropy

- 설명: ont-hot-encoding을 쓸경우, 출력차원이 크면 많은 메모리를 필요로 한다.  
       (ex: 100의 출력 * 입력data 100만건이상, colab에서 20G이상 소진됨)
       이때 label을 정수형으로 변환가능하면, one-hot 대신 integer-type loss함수를 쓸수 있다.  
       그럼 one-hot보다 훨씬 작은 메모리 사용이 가능

- 참고1 : https://crazyj.tistory.com/153  
- 참고2 : https://www.dlology.com/blog/how-to-use-keras-sparse_categorical_crossentropy/  

- sample code
```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy', #categorical_crossentropy -> sparse_categorical_crossentropy
              metrics=['sparse_categorical_accuracy']) #accuracy -> sparse_categorical_accuracy
``


- one-hot-encoding shape 맞출때(3차원->2차원) 적용방안
```
.. 3차원
y_hat = GlobalAveragePooling1D()(y_hat) #2차원
..
```
