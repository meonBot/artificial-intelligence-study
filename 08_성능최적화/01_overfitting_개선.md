# 오버피팅(과적합) 문제 해결방안

- 해결방안:  regularizaiton(3가지), early-stop, droup-out 

### 1. regularizaiton

#### 1-1. 일반 Layer
- https://keras.io/ko/layers/core/ : 주로 kernel_initializer, bias_initializer 사용  
```
kernel_initializer: kernel 가중치 행렬의 초기값 설정기 (초기값 설정기를 참조하십시오).
bias_initializer: 편향 벡터의 초기값 설정기 (초기값 설정기를 참조하십시오).
kernel_regularizer: kernel 가중치 행렬에 적용되는 정규화 함수 (정규화를 참조하십시오).
```
- https://light-tree.tistory.com/125 : L1, L2 regulirization 비교  
>L1 Norm 은 파란색 선 대신 빨간색 선을 사용하여 특정 Feature 를 0으로 처리하는 것이 가능하다고 이해할 수 있습니다.   
>다시 말하자면 L1 Norm 은 Feature selection 이 가능하고 이런 특징이 L1 Regularization 에 동일하게 적용 될 수 있는 것입니다.   
>이러한 특징 때문에 L1 은 Sparse model(coding) 에 적합합니다. L1 Norm 의 이러한 특징 때문에 convex optimization 에 유용하게 쓰인다고 합니다.  

#### 1-2. Embedding Layer
- embeddings_regularizer=regularizers.l2(0.001)
```
embedding_layer = keras.layers.Embedding(len(gd_data.input_vocap.tokenizer.word_index) + 1,   #in  dim
                            EMBEDDING_DIM,                                #out dim
                            embeddings_initializer='glorot_uniform',      #glorot_uniform: Xavier uniform initializer.
                            embeddings_regularizer=regularizers.l2(0.001),
                            weights=[embedding_matrix],
                            input_length=MAX_PADDING,
                            trainable=True) # trainable=True
```

### 2. early-stop

#### 2-1. model.fit 함수에 적용
```
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  #loss='binary_crossentropy', sparse_categorical_crossentropy, categorical_crossentropy
              metrics=['acc'])

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping()

history = model.fit(gd_data.x_train_data, #gd_data.inputs_tr, dd
                    gd_data.targets_tr, #gd_data.targets_tr,
                    epochs=30,
                    batch_size=1024,
                    validation_split=0.1,              #validation data를 트레이닝 data에서 분할하여 사용하고 셔플해서 씀
                    shuffle=True,
                    verbose=1 # verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
                    ,callbacks=[early_stopping])
````

#### 2.2 ModelCheckpoint + fit_generator에 적용
```
cp = ModelCheckpoint("./weights/NMT.{epoch:02d}-{val_loss:.2f}.hdf5",
                     monitor='val_loss',
                     verbose=0,
                     save_best_only=True,
                     save_weights_only=True,
                     mode='auto')
                     
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping()

start = time()
print(' Ctrl+C to end early.')
try: model.fit_generator(generator=gd_data.generator(BAT_SIZE_TEMP, 'TR'), 
                        steps_per_epoch = TR_STEP_EPOCH,
                        validation_data=gd_data.generator(VA_BAT_SIZE_TEMP, 'VA'),
                        validation_steps = VA_STEP_EPOCH,
                        callbacks=[early_stopping, cp],
                        workers=1,
                        verbose=1,
                        epochs=5) 
```

### 3. drop-out
```
from keras.layers import Dropout
model.add(keras.layers.Dense(1024 ,activation="relu" ))
model.add(Dropout(0.2))
```
