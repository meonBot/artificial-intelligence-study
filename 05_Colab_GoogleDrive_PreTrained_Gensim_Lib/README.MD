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

### 8. Colab Pro (Colab 유로버전)
- jupyter notebook만 사용하는경우, GCP 서버 설치할 필요없이, 유로버전으로 싼가격으로 빠르게 작업가능!
![image](https://user-images.githubusercontent.com/45334819/74098945-551b8800-4b61-11ea-9ff6-bbb56ae525a1.png)
![image](https://user-images.githubusercontent.com/45334819/74098948-59e03c00-4b61-11ea-80b5-7a8222d3935f.png)
![image](https://user-images.githubusercontent.com/45334819/74098949-5e0c5980-4b61-11ea-9d21-c8b5ee738058.png)

### 9. Colab Download File-Size occurs error
> 간혹, pre-trained data(6G)를 여러번 다운로드 하는경우(하루 최대 70~80G?) IO오류가 발생함  
> 다른 계정으로 진행시 문제없음  
![image](https://user-images.githubusercontent.com/45334819/74835457-bfea7180-5360-11ea-8ad1-4d3b7f8d06d4.png)  
