# 최소분산을 최대가중치로 변환하는 python코드

- 설명 : 
> k-means 알고리즘에서 DATA가 K중심값에 몰려있는것이 좋다.   
> 이때 분산값을 최소화 해야하는데, 분산이 최소가 될때 가중치가 가장크게 반영되도록 변환이 필요했다.  
> 하여 기초수학(평균,분산,표준편차)과 고등수학(로그함수, 지수함수)를 학습한뒤 아래와 같은 샘플코드를 작성하였다.  

```
import math
import numpy as np
#분산 최소값을 최대값으로 변환하기 위한 로그함수 활용방법
#참고(고등수학): https://www.youtube.com/watch?v=I_H04p9HHcI

#1. 거리 평균 (정규화했다고 가정)
mean_X = [0.1, 0.4, 0.5]
Y = [0,0,0]
print('mean_X:', mean_X)

#2. 분산(편차의 제곱의 평균 : 변수가 0과 X라고 보면, 분산은 편차(X/2)의 제곱이다)
X = np.array(mean_X)
var_X = [math.pow(val/2, 2) for val in X]     
print('var_X:', var_X)

#3.log함수(밑2) -> x축 회전(밑2->1/2(X축회전)->0.5)
log_X = [math.log(val, 0.5) for val in var_X]
print('log_X:', log_X)

#4.정규화
sum_X = sum(log_X)
nor_X = [val/sum_X for val in log_X]
print('nor_X:', nor_X)

#mean_X: [0.1, 0.4, 0.5]
#var_X: [0.010000000000000002, 0.16000000000000003, 0.25]
#log_X: [6.643856189774724, 2.6438561897747244, 2.0]
#nor_X: [0.588591910067779, 0.2342242697966631, 0.17718382013555792]

import matplotlib.pyplot as plt
plt.plot(mean_X)
plt.plot(nor_X)
plt.show()
```
