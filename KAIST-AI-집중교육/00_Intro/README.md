# AI 개발환경 세팅
### anaconda, tensorflow, python 설치



1)anaconda설치(python내장됨)
>https://www.anaconda.com/distribution/#download-section

2)gpu사용관련 CUDA등 설치(NVIDA없어서 생략)

3)anaconda prompt 관리자 권한실행 > 아래 명령어 실행
① python –m pip install --upgrade pip 
② conda create –n <가상환경 이름> python=3.7 
③ conda activate <가상환경 이름> 
④ pip install tensorflow   (gpu사용시  tensorflow-gpu)
⑤ conda install matplotlib 
⑥ pip install keras 
⑦ conda install jupyter notebook

4) Jupyter(Machine Learning IDE) 실행방법
4-1) local에 설치한 Jupyter 실행파일로 실행 (Jupyter Notebook (tensorflow))
4-2) Jupyter 프로젝트 파일이 존재하는 탐색기에서, 주소에 "jupyter notebook"입력
4-3) 구글 cloud jupyter 실행(https://colab.research.google.com/)

5) Python(Tensorflow용) 실행방법
>Anaconda Prompt 실행
>conda activate tensorflow(가상환경 설치 이름)
>python .py파일명
