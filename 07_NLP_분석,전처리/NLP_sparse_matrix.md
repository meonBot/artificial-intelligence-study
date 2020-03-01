# 

### 참고 : https://docs.scipy.org/doc/scipy/reference/sparse.html

```
import numpy as np
from scipy.sparse import csr_matrix

A = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
print('type(A):', type(A))
# type(A): <class 'scipy.sparse.csr.csr_matrix'>
print('A:', A)
#A: (0, 0)	1
#(0, 1)	2
#(1, 2)	3
#(2, 0)	4
#(2, 2)	5

v = np.array([1, 0, -1])
A_dot = A.dot(v)
print('A.dot(v):', A_dot)
#A.dot(v): [ 1 -3 -1]
```
