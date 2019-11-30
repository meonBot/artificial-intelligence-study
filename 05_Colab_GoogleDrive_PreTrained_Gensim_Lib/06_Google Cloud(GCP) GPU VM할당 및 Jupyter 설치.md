# 06_Google Cloud(GCP) GPU VM할당 및 Jupyter 설치.md

<br>
- 참고: https://jeinalog.tistory.com/entry/GCP-%EB%94%A5%EB%9F%AC%EB%8B%9D%EC%9D%84-%EC%9C%84%ED%95%9C-VM-%EC%9D%B8%EC%8A%A4%ED%84%B4%EC%8A%A4-%EA%B5%AC%EC%84%B1-feat-GPU  

*** 2019년11월말 현재 모든 gpu가 초과사용상태이므로, Virtual GPU를 할당받아 사용하도록 함.   
*** OS는 우분투 16.04, 아나콘다는 설치하지 않음   
*** GPU사용을 위해서는 가상환경으로 구동되어야함   

<br>

## 설치  

#### 1.패키지 설치
- pip3
```
sudo apt-get update
sudo apt-get install python3-pip -y
```

#### 2.CUDA 설치
- 아래 명령어로 먼저 설치파일을 다운받아주세요.
```
curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
sudo apt-get update
sudo apt-get install cuda-9-0
```
- 확인: nvidia-smi

### 3.cuDNN 설치
- 역시 먼저 아래 명령어로 설치파일을 다운받습니다.
```
sudo sh -c 'echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" >> /etc/apt/sources.list.d/cuda.list'
sudo apt-get update
sudo apt-get install libcudnn7-dev
```
#### 4. 기타 라이브러리 설치
- pip3 버전확인
```
pip3 --version
>pip 8.1.1 from /usr/lib/python3/dist-packages (python 3.5)
```
- python 3.6으로 올리기(jupyter 오류발생으로)
- https://unipro.tistory.com/237
- Ubuntu 16.04에는 써드파티 PPA를 추가해야 python 3.6을 설치할 수 있다.
```
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt-get update
sudo apt-get install python3.6
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
```
- 확인 python3 -V  //3.5.2->3.6.8로 바뀌어있음

#### 5. 가상환경 설치 
*** gpu사용을 위한 가상환경 추가
- 가상환경 유틸 설치
```
pip3 install ipykernel
sudo pip3 install virtualenv
- 가상환경 추가
virtualenv keras_gpu --python=python3.6
- 가상환경 사용
source keras_gpu/bin/activate
python -m ipykernel install --user --name keras_gpu
- 가상환경에서 keras등 설치하고 실행
pip3 install tensorflow-gpu keras //torch torchvision
pip3 install jupyter sklearn pandas //matplotlib seaborn (설치오류는 일단 두개는 제외)
```

#### 6. jupyter notebook 사용하기
```
jupyter notebook --generate-config
vi ~/.jupyter/jupyter_notebook_config.py
```
- I로 insert(편집) 모드로 전환한 후 다음 세 설정만 수정해주면 됩니다.  
- config file의 모든 코드들은 현재 주석처리 되어있으므로 그냥 맨 위나 맨 밑에 아래 코드를 추가하면 됩니다.  
ㅤ
- [Example]
```
c = get_config()
c.NotebookApp.ip = '35.247.25.58'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
```
- 35.247.25.58 위치에는 GCP에서 할당받은 외부 IP를 입력합니다.  
- 8888은 우리가 방화벽을 만들 때 설정했던 포트입니다.  
- 접속: 
> jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root 

- 웹 브라우저 상에서 34.82.13.104:8888 의 주소 형태로 ip를 알맞게 입력하면 주피터 노트북에 접속할 수 있습니다.  
- 최초 실행 시 다음과 같은 화면이 뜹니다.  

- GPU 모니터링: 
```
watch -d -n 0.5 nvidia-smi
```

