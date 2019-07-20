

# Colab with Google Drive Tip
<br>

## 1. File Upload & Download
<br>

- 참고 : http://www.dreamy.pe.kr/zbxe/CodeClip/3769485

### 1-1. Colab 파일 업로드:

#### 1-1. 직접 업로드(files.upload())
```python
from google.colab import files
#파일업로드창 출력
uploaded = files.upload()
#업로드한 파일정보 출력
for fn in uploaded.keys():
  print('Upload file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))
#읽어서 사용
im = imread("output.jpg")
```

#### 1-2.google drive업로드 후 로딩 방법

##### 1-2-1. File Open 및 Line단위 처리
``` python
embeddings_index = {}
# driver code
from google.colab import drive
import os
drive.mount('/content/gdrive') # 인증필요

f = open(os.path.join('/content/gdrive/My Drive', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))
```

##### 1-2-2. Gensim 및 dataPath 이용하여 로딩
``` python
from gensim.test.utils import datapath, get_tmpfile
from google.colab import drive
drive.mount('/content/gdrive')

import smart_open
glove_file = datapath('/content/gdrive/My Drive/glove.6B.100d.txt') #/content/gdrive/My Drive/NLP-Lab/glove.6B/glove.6B.100d.txt
word2vec_glove_file = get_tmpfile("glove.6B.100d.word2vec.txt") #temp파일생성
glove2word2vec(glove_file, word2vec_glove_file)      #로딩한 파일을 temp파일로 옮김?
xy = np.loadtxt( glove_file , delimiter=',', dtype=np.float32) 
```
##### 1-2-3. google drive 이용 File path로 접근하는 방법

#### ex) image
``` python
from google.colab import drive
drive.mount('/content/gdrive')
from matplotlib.pyplot import imread

im = imread("/content/gdrive/My Drive/NLP-Lab/output.jpg")
```

##### ex) txt
``` python
from google.colab import drive
drive.mount('/content/gdrive')

#csv data read
xy = np.loadtxt('/content/gdrive/My Drive/data-01-test-score.csv', delimiter=',', dtype=np.float32) 
#문자열인 경우 dtype=np.str
#google driave root 경로: /content/gdrive/My Drive/
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
#단 chunk_size 초과로 오류발생 가능..
```
<br>

### 1-2. Colab -> Local 다운로드(파일 저장) 방법

#### 1-2-1. Pandas & file download (문자열?)
``` python
import pandas as pd 
pd.DataFrame(err_vec).to_csv("/tmp/preporcessing_case6.csv")

from google.colab import files
files.download("/tmp/preporcessing_case6.csv")
```

#### 1-2-2. Numpy download(숫자?)
``` python
import numpy
a = numpy.asarray([ [1,2,3], [4,5,6], [7,8,9] ])
numpy.savetxt("foo.csv", a, delimiter=",")

from google.colab import files
files.download("파일명")
```
<br>

<hr />
<br>

## 2. Tokenizer, Trained Model 저장 및 복원 방법 (Colabb <-> Google Drive)

- 참고: https://yamalab.tistory.com/80

### 2-1. Trained Model 저장, 복원

#### 2-1-1. Colab -> Google Drive 저장
``` python
!pip install -U -q PyDrive

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# 1. Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# PyDrive reference:
# https://googledrive.github.io/PyDrive/docs/build/html/index.html

# 2. Create & upload a file text file.
# 특정 폴더 안으로 파일 삽입
uploaded = drive.CreateFile({'title': 'GoodsDesc_Validation_20190629_NEW130.h5'}) #, "parents": [{"kind": "drive#fileLink","id": 'google'}]})
uploaded.SetContentString('GoodsDesc_Validation_20190629_NEW130')
uploaded.SetContentFile('GoodsDesc_Validation_20190629_NEW130.h5')
uploaded.Upload()
print('Uploaded file with ID {}'.format(uploaded.get('id')))

# 3. Load a file by ID and print its contents.
downloaded = drive.CreateFile({'id': uploaded.get('id')})
```

#### 2-1-2. Googe Drive -> Colab 복원(로딩)
``` python
#구글 드라이브에서 다운로드
from google.colab import auth
auth.authenticate_user()

from googleapiclient.discovery import build
drive_service = build('drive', 'v3')

import io
from io import BytesIO   
from googleapiclient.http import MediaIoBaseDownload

#https://drive.google.com/open?id=1IPrvTIiicBXt3_gESAAdAdzhm8v81xv2
request = drive_service.files().get_media(fileId='1IPrvTIiicBXt3_gESAAdAdzhm8v81xv2')
downloaded = io.BytesIO()
downloader = MediaIoBaseDownload(downloaded, request)
done = False
while done is False:
  status, done = downloader.next_chunk()
  if status:
      print("Download %%%d%%." % int(status.progress() * 100))
  print("Download Complete!")

downloaded.seek(0)

with open('/tmp/GoodsDesc_Validation_20190629_NEW130.h5', 'wb') as f:
    f.write(downloaded.read())

from keras.models import load_model
model = load_model('/tmp/GoodsDesc_Validation_20190629_NEW130.h5')
```

### 2-2. Tokenizer 저장 및 복원

#### 2-2-1. Colab File Upload -> Pickle Loading
```python
from google.colab import files
#파일업로드창 출력
uploaded = files.upload()  # 파일명:tokenizer.pickle

#업로드한 파일정보 출력
for fn in uploaded.keys():
  print('Upload file "{name}" with length {length} bytes'.format(name=fn, length=len(uploaded[fn])))
  
import pickle
# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
```

#### 2-2-2. Pickle saving -> Colab File Download
```python
import pickle
# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#다운로드: 
from google.colab import files
files.download("tokenizer.pickle")
```

- pickle 예제: https://wikidocs.net/8929  

#### 2-2-3. Json Format 예제
``` python
# tokenizer의 내용을 json으로 받아서 디스크에 저장
text_to_save = tokenizer.to_json()
# Json file -> File download

# 디스크에 저장된 json을 읽어서 tokenizer_from_json로 지정해서 tokenizer 생성
from keras_preprocessing.text import tokenizer_from_json
# File Upload -> Json load
tokenizer = tokenizer_from_json(text_to_save)
```

