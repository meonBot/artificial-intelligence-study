# 성능 최적화

<br>

### 1. [Overfitting 해결](https://github.com/jukyellow/artificial-intelligence-study/blob/master/08_%EC%84%B1%EB%8A%A5%EC%B5%9C%EC%A0%81%ED%99%94/01_overfitting_%EA%B0%9C%EC%84%A0.md)

### 2. [One-Hot-Encoding 메모리 문제해결, sparse_categorical_crossentropy](https://github.com/jukyellow/artificial-intelligence-study/blob/master/08_%EC%84%B1%EB%8A%A5%EC%B5%9C%EC%A0%81%ED%99%94/02_sparse_categorical_crossentropy.md)  

### 3. [성능평가(keras, sklearn)](https://github.com/jukyellow/artificial-intelligence-study/blob/master/08_%EC%84%B1%EB%8A%A5%EC%B5%9C%EC%A0%81%ED%99%94/03_keras_metrics_performance_eval_ver2_0_20200619.ipynb)

### 4. 앙상블

#### 4-1. Bagging(랜던포레스트),Boosting(그레디언트부스팅)

#### 4-2. 모델 Averaging

### 5. Imbalance data 처리
#### 5-1. Data Normailzation(Over/Under Sampling)
#### [5-2. Focal-loss](https://github.com/jukyellow/artificial-intelligence-study/blob/master/08_%EC%84%B1%EB%8A%A5%EC%B5%9C%EC%A0%81%ED%99%94/05_2_keras_focal_loss_test_20200709.ipynb)
<br>

### 6. Data Normalization  
#### 6-1. OverSampling, UnderSampling  
#### 6-2. MinMaxSacaler  

### 7. Weight handling
#### 7-1. class_weight  
#### 7-2. sample-weight

### 8. K-Fold Validation
<br>

### 9. Cross-Entropy loss  
- https://wikidocs.net/71597, 
- https://www.tensorflow.org/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy?hl=ko
```
import tensorflow as tf
loss = tf.keras.losses.SparseCategoricalCrossentropy()
loss_value = loss(y_data, predictions)
#scce(y_true, y_pred).numpy() -> ex) 1.256
```
