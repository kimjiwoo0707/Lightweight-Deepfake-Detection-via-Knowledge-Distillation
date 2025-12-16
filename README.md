# 딥페이크 탐지를 위한 지식 증류 기반 경량화 모델


---

## 📰 연구 배경
딥페이크(Deepfake)는 AI를 이용해 이미지, 영상, 음성을 조작하여 실제와 유사한 콘텐츠를 만드는 기술로 최근 오픈소스 프로그램 확산으로 누구나 쉽게 제작 가능해지면서 사회적·윤리적 문제가 증가하고 있다.  
경찰청 통계에 따르면, 딥페이크 관련 범죄는 2021년 156건에서 2024년 7월 기준 297건으로 증가했다. 특히 성 착취물 제작 등 악용 사례가 문제이며, 유포된 콘텐츠는 완전 삭제가 어려워 2차·3차 피해로 이어질 수 있다.  
본 연구는 **경량화된 모델**과 **지식 증류(Knowledge Distillation)** 기법을 활용해 딥페이크 얼굴 이미지를 실시간으로 탐지하고, 피해 확산을 줄이는 것을 목표로 한다.

---

## 📰 데이터 구성

*Kaggle – Deepfake-dataset (140k+dataset real or fake)*    [데이터 링크](https://www.kaggle.com/datasets/tusharpadhy/deepfake-dataset)

### 데이터셋 개요
- **총 이미지 수**: 331,335개의 이미지
- **크기**: 약 6GB  
- **구성**: 실제(real) 딥페이크(fake)로 구분된 이미지 데이터

### 데이터셋 특징
- 다양한 얼굴 이미지: 실제 인물과 딥페이크 얼굴 이미지 포함  
- 다양한 딥페이크 생성 기법으로 제작되어, 모델 일반화 능력 향상에 유리  
- 라벨링 정보: `real` 또는 `fake`

### 데이터 전처리
1. **크기 조정(Resize)**: 모든 이미지를 256x256 픽셀로 조정  
2. **중앙 자르기(Center Crop)**: 얼굴 중심을 기준으로 224x224 픽셀로 자르기  
3. **텐서 변환 및 정규화**: PyTorch `transforms`로 텐서 변환, 픽셀 값 [0,1] 범위 정규화  
4. **표준화(Normalization)**: ImageNet 평균값 `[0.485, 0.456, 0.406]`과 표준편차 `[0.229, 0.224, 0.225]` 사용
5. train, validation, test 각각 7:2:1로 분할
---

## 📰 모델 구조

### Teacher 모델
- EfficientNet-b7 사용  
- 대형 CNN, 수백 레이어, 수백만 파라미터  
- 복잡한 딥페이크 특징 학습에 적합  

### Student 모델 (ResNet8)
- 경량화 모델, 파라미터 0.145M (Teacher 대비 99.73% 감소)  
- 입력: 3채널 이미지  
- Conv2d → 16 출력 채널  
- BasicBlock: 2 x 3x3 Conv + BatchNorm + ReLU  
- Shortcut Connection: 입력과 출력 합산 → 기울기 소실 방지  
- 총 28개 레이어

---

## 📰 지식 증류 (Knowledge Distillation)

Teacher 모델의 **Soft label**과 Ground truth **Hard label**을 결합하여 Student 모델 학습  

<img width="742" height="486" alt="image" src="https://github.com/user-attachments/assets/abe7618c-9061-4fbb-b566-5ca81bdd9172" />


### 손실 함수
L= α ∙ KL(T,S)+(1-α)  ∙ L_hard 
L= α ∙ MSE(T,S)+(1-α)  ∙ L_hard   
- Soft Loss: Teacher ↔ Student 출력 분포 정렬 (KL / MSE)  
- Hard Loss: True labels와 차이 최소화  
- α: Soft / Hard Loss 가중치 조절 [10]

---

## 📰 모델 성능 평가

### Teacher 모델 성능 [모델 코드](https://github.com/kimjiwoo0707/Lightweight-Deepfake-Detection-via-Knowledge-Distillation/blob/6f42fa36af42271fdccc12d009b8c0e38b27ab95/Teacher_Model.py)

#### Teacher 모델 (최종 성능)  

EfficientNet-b7 기반 Teacher 모델의 최적 설정 결과  
| 모델 | Accuracy(%) | F1-score(%) | FPS | params(M) |
|-------|------------|-------------|-----|----------|
| EfficientNet-b7 | 96.87 | 96.81 | 204.13 | 65.1 |

### Student 모델 성능 (ResNet8) 및 지식 증류 효과 [모델 코드](https://github.com/kimjiwoo0707/Lightweight-Deepfake-Detection-via-Knowledge-Distillation/blob/6f42fa36af42271fdccc12d009b8c0e38b27ab95/Student_KD.py)

| 모델 | Accuracy(%) | F1-score(%) | FPS |
|-------|------------|-------------|-----|
| ResNet8 (no KD) | 86.70 | 87.31 | 617.72 |
| ResNet8 (with KD) | 88.63 | 88.35 | 617.72 |

지식 증류 적용 시, 동일한 학습 설정에서 Accuracy가 약 2% 향상되었으며, Teacher 모델 단독 추론 대비 약 3배 높은 FPS를 달성하였다.  
본 비교는 KD 적용 효과를 확인하기 위한 예비 실험으로, 동일한 학습 설정에서 KD 적용 여부만을 on/off 하여 평가하였다. 이후 하이퍼파라미터 및 손실 함수 탐색을 통해 최종 성능을 추가로 개선하였다.

#### 지식 증류 적용 후 ResNet8의 하이퍼파라미터 조정 및 성능 비교

| Batch Size | Learning Rate | Scheduler          | α   | Accuracy (%) | F1-score (%) |
|------------|---------------|------------------|-----|--------------|--------------|
| 32         | 0.00001       | -                | 0.5 | 92.36        | 92.05        |
| 32         | 0.00001       | -                | 0.6 | 92.32        | 91.99        |
| 32         | 0.0001        | -                | 0.5 | 92.74        | 92.51        |
| 32         | 0.0001        | ReduceLROnPlateau | 0.4 | 93.92        | 93.85        |

#### 최종 Student 모델 성능 (지식 증류 적용)

| Loss Type       | Accuracy (%) | F1-score (%) | FPS     | Params (M) |
|-----------------|--------------|--------------|---------|------------|
| KL Divergence   | 93.92        | 93.85        | 617.72  | 0.145      |
| MSE             | 94.29        | 94.26        | 617.72  | 0.145      |

KL Divergence 기반 KD 결과는 상기 하이퍼파라미터 탐색에서 도출된 최적 설정과 동일한 조건에서 평가하였다. 동일한 설정에서 distillation loss를 MSE로 변경한 경우, Accuracy와 F1-score가 추가적으로 향상되는 것을 확인하였다.

- MSE 기반 KD 적용 시 테스트 Accuracy 94.29% 달성
- Teacher 모델 대비 Accuracy 차이: 2.58%  
- FPS: 617.72 (Teacher 모델 단독 추론 대비 약 3배)
- Params: 지식증류 적용 후 → Teacher 대비 약 99.78% 감소
  
> 결과적으로 지식 증류를 통해 대폭 경량화된 Student 모델에서도, distillation loss를 MSE로 적용함으로써 성능 저하를 최소화하면서 높은 분류 성능과 실시간 추론이 가능한 수준의 속도를 동시에 확보할 수 있음을 확인하였다.

---
## 📰 한계

- 본 연구에서 사용한 딥페이크 데이터셋은 단일 방식으로 생성된 이미지로 구성됨, 따라서 다른 방식이나 새로운 딥페이크 생성 기법으로 제작된 데이터에서는 모델 성능이 떨어질 가능성이 있음  
- 향후 연구에서는 다양한 생성 기법과 출처의 데이터로 모델 일반화 능력을 평가할 필요가 있음
