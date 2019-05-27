(base) C:\Users\jukye>activate tensorflow

(tensorflow) C:\Users\jukye>python
Python 3.7.3 (default, Mar 27 2019, 17:13:21) [MSC v.1915 64 bit (AMD64)] :: Anaconda, Inc. on win32
Type "help", "copyright", "credits" or "license" for more information.

>>> import tensorflow as tf
>>> tf.__version__
'1.13.1'

>>> hello = tf.constant("Hello, Tensorflow")
>>> sess = tf.Session()
2019-05-11 23:57:25.937105: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
>>> print(sess.run(hello))
b'Hello, Tensorflow'
