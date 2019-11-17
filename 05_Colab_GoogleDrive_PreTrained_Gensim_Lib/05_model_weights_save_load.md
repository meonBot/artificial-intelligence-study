## 모델 및 weight 저장/로딩 및 Unkown Layer(사용자정의 클래스) 해결방법

### 1. Model, Weight Save
#### 1-1. weight save
```
import os
from keras.callbacks import ModelCheckpoint

if not os.path.exists('./weights'):
    os.makedirs('./weights/')

cp = ModelCheckpoint("./weights/NMT.{epoch:02d}-{val_loss:.2f}.hdf5",
                     monitor='val_loss',
                     verbose=0,
                     save_best_only=True,
                     save_weights_only=True,
                     mode='auto')
                     
history = model.fit(x_train_data, 
                    y_train_data,
                    epochs=3,
                    batch_size=512,
                    validation_split=0.1,              
                    shuffle=True,
                    callbacks=[cp],        #callabck 함수로 weight save
                    verbose=1) # verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
```
#### 1-2. model save
```
if not os.path.exists('./models'):
    os.makedirs('./models/')

model.save('./models/cls_att_20191117.h5') # 모델 저장
```

### 2. Model, Weight loading
```
from keras.models import load_model

HERE = '/content'
SAMPLE_WEIGHTS = os.path.join(HERE, 'weights', 'NMT.03-0.14.hdf5')
weights_file = os.path.expanduser(SAMPLE_WEIGHTS)

# 모델 불러오기 -> Known Layer발생시(사용자 정의 class 사용하는경우) -> custom_objects로 이름 추가
test_model = load_model('./models/cls_att_20191117.h5', custom_objects={'AttentionWithContext':AttentionWithContext})
# 모델을 로딩하지 않고, 모델은 생성하고 weight는 로딩하는경우 weight값이 제대로 로딩되지 않음(이유는 확인못함)

test_model.load_weights(weights_file, by_name=True)
test_model.compile(optimizer='adam', loss='categorical_crossentropy')
#->predict...
```

