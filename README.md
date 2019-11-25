# 자막 기반 영상 분류 시스템 구축

## Getting Started

EBS 제공 유튜브 영상의 영역과 시청 연령 분류 모델 개발
- EBS 사에서 영역과 난이도, 연령, 지능으로 분류한 학습용 영상과 키워드 구축
- 학습용 데이터가 8000 여 건으로 제한적

### 개요

####학습
1. 기존 EBS 사에서 영역별 정의한 영상의 키워드를 기반으로 TF-IDF(빈도) 와 LDA(출현 확률 군집화) 특징 추출

2. 영상의 자막 데이터에 대한 그래프 모델을 정의하고 키워드를 추출
 
3. 키워드에 대한 Word2Vec, 자막에 대한 Elmo 를 적용한 특징 추출
 
 
 
####분류

- 첫번째, TF-IDF 와 W2V 특징에 기반한 코사인 유사도 분류 

- 두번째, 선형 회귀, SVM 등 머신러닝과 DNN, CNN 딥러닝 분류기 적용

- 세번째, 각 특징 추출 결과에 대한 키워드 임의 가중치 조절과 퓨샷 러닝 기법 적용



####분류 성능 평가
머신러닝 우수 모델
- 각 영역에 대한 평균 accuracy 87%, precision 85%, recall 81%, f1score 83%

딥러닝 우수 모델
- 각 영역에 대한 평균 accuracy 75%, precision 78%, recall 73%,f1score 75%

### Installing

```
pandas 0.23.0

sklearn 0.19.1
```

## 공용 데이터 셋

공용 데이터 셋
- 한국어 : https://www.kaggle.com/bryanpark/korean-single-speaker-speech-dataset
- 영어 : http://www.openslr.org
        : http://voxforge.org/

