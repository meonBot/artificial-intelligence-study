# Colab & Google Drive & Gensim & Pre-Trained Model Lib
<br>

### 1. Colab & Google Drive (File Upload/Download, Model/Tokenizer Save/Load)
- https://github.com/jukyellow/artificial-intelligence-study/blob/master/05_Colab_GoogleDrive_PreTrained_Gensim_Lib/01_Colab_FileUpDown_By_GoogleDrive.md

### 2. Gensim

#### 2-1. Gensim 예제

#### 2-2. Gensim & FastText Loading & Visualization
- https://github.com/jukyellow/artificial-intelligence-study/blob/master/05_Colab_GoogleDrive_PreTrained_Gensim_Lib/02_02_Gensim_FastText_Loading_Visualization.ipynb

### 3. NLP(자연어 처리) 및 & Pre-Trained Model

#### 3-1. Pre-Tranied Model

#### 3-1-1. Glove
- https://github.com/jukyellow/artificial-intelligence-study/blob/master/05_Colab_GoogleDrive_PreTrained_Gensim_Lib/03_01_Glove_Loading_Gen_Weights.md

#### 3-1-2. FastText  


### 4. Google Colab & Github 소스연동(.py파일 로딩 및 호출) 방법
- https://github.com/jukyellow/artificial-intelligence-study/blob/master/05_Colab_GoogleDrive_PreTrained_Gensim_Lib/04_colab_github_clone_use.py  


### 5. Model, Weights 저장/로딩 및 Unknown Layer(사용자정의 클래스) 해결방법
- https://github.com/jukyellow/artificial-intelligence-study/blob/master/05_Colab_GoogleDrive_PreTrained_Gensim_Lib/05_model_weights_save_load.md
  

### 6. Google Cloud(GCP) GPU VM할당 및 Jupyter 설치
- gpu limit가 전세계적으로? 초과되어서 virtual gpu로 신청하면 가능  
- 64G(16*4, P100모델) 할당!(300 무로 크레딧으로 5일정도 사용가능)  
https://github.com/jukyellow/artificial-intelligence-study/blob/master/05_Colab_GoogleDrive_PreTrained_Gensim_Lib/06_Google%20Cloud(GCP)%20GPU%20VM%ED%95%A0%EB%8B%B9%20%EB%B0%8F%20Jupyter%20%EC%84%A4%EC%B9%98.md 

### 7. Google Colab, Timeout 없이 12시간 연속사용하기
![image](https://user-images.githubusercontent.com/45334819/71640035-1e329680-2cc6-11ea-8ed7-507c197cde71.png)
- Ctrl+Shift+i (개발자환경F12?)열고 console tab에 아래 스크립트 추가 : 60초마다 refresh되어 timeout 발생안함  
```
function ClickConnect(){
  console.log("Working"); 
  document.querySelector("colab-toolbar-button#connect").click() 
}
setInterval(ClickConnect,60000)
```
- 해당 기능이, 아래와 같이 오류가 발생 사용할수 없음(2020년 1분이 어느시점에서 부턴가)
```
Skipping recording uncaught error from XHR: unknown name [object ErrorEvent]
(anonymous) @ external_polymer_binary_l10n__ko.js?vrz=colab-20200417-085601-RC00_307029476:1551
VM2494:3 Uncaught TypeError: Cannot read property 'click' of null
    at ClickConnect (<anonymous>:3:57)
```
- 다른 해결책(마우스 자동클릭 방식): mouse_auto_click.py (참고: https://stackoverflow.com/questions/54057011/google-colab-session-timeout#new-answer)
  를 local(window등)환경에서 실행하고, s클릭시 마우스 포인트의 좌클릭이 시작됨

### 8. Colab Pro (Colab 유로버전)
- jupyter notebook만 사용하는경우, GCP 서버 설치할 필요없이, 유로버전으로 싼가격으로 빠르게 작업가능!
![image](https://user-images.githubusercontent.com/45334819/74098945-551b8800-4b61-11ea-9ff6-bbb56ae525a1.png)
![image](https://user-images.githubusercontent.com/45334819/74098948-59e03c00-4b61-11ea-80b5-7a8222d3935f.png)
![image](https://user-images.githubusercontent.com/45334819/74098949-5e0c5980-4b61-11ea-9d21-c8b5ee738058.png)

### 9. Colab Download File-Size occurs error
> 간혹, pre-trained data(6G)를 여러번 다운로드 하는경우(하루 최대 70~80G?) IO오류가 발생함  
> 다른 계정으로 진행시 문제없음  
![image](https://user-images.githubusercontent.com/45334819/74835457-bfea7180-5360-11ea-8ad1-4d3b7f8d06d4.png)  

### 10. Colab tensorflow 2.x 오류 -> 1.x downgrade 및 적용방법
- 오류 캡쳐:
![image](https://user-images.githubusercontent.com/45334819/77914903-96aef080-72d1-11ea-9e55-2b75caab86fe.png)
- downgrade 방법:
```
%tensorflow_version 1.x
import tensorflow

#!pip install tensorflow==1.14.0 # 이렇게 직접 2.x uninstall 및 1.14 install하는 방식은 여전히 오류존재
```

### 11. Colab -> GCP Upload 오류 해결방안

- 오류상세 : RedirectMissingLocation: Redirected but the response is missing a Location: header.
- 버전 확인
> !pip list  
> httplib2                 0.17.1  
> google-api-python-client 1.7.12    

- 참고 : https://github.com/tokland/youtube-upload/issues/293
- 해결방안 : 아래와 같이 특정 lib를 삭제하고, downgrade하여 재설치  
```
!pip uninstall google-api-python-client -y
!pip uninstall oauth2client -y
!pip uninstall progressbar2 -y
!pip uninstall httplib2 -y
!pip install google-api-python-client==1.7.3
!pip install oauth2client==4.1.2
!pip install progressbar2==3.38.0
!pip install httplib2==0.15.0
```
<br>

### 12. Colab Error[input/output Error] 해결방법
-오류 : OSError: [Errno 5] Input/output error  
- 해결방안: google drive 파일 삭제  
![image](https://user-images.githubusercontent.com/45334819/88229290-743f4980-ccab-11ea-8f2b-40ca512e76fc.png)
<br>

### 13. Colab Memory 35G까지 늘리는 방법
- Colab pro(고용량 RAM: 25G) -> 런타임유형 변경(TPU: 35G !!!)  
<br>
