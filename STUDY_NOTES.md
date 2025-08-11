# SegRNN 학습 포인트 정리

## 논문 핵심 이해 포인트

### 1. Segmentation 전략의 혁신성
- **문제점**: 기존 RNN은 point-wise iteration으로 긴 시퀀스 처리 시 메모리와 계산 비용 증가
- **해결책**: Segment-wise iteration으로 시퀀스를 segment 단위로 분할
- **효과**:
  - 메모리 복잡도: O(L) → O(L/W)
  - Vanishing gradient 문제 완화
  - 장기 의존성 학습 능력 향상

### 2. PMF vs RMF 디코딩 전략
#### RMF (Recurrent Multi-step Forecasting)
- 순차적으로 각 segment 예측
- Autoregressive 특성 유지
- 이전 예측이 다음 예측의 입력이 됨

#### PMF (Parallel Multi-step Forecasting)
- 모든 segment를 병렬로 예측
- 위치 임베딩으로 시간 정보 구분
- 속도 향상 및 error accumulation 감소

### 3. 단일 레이어의 위력
- 복잡한 다층 구조 없이 단일 GRU 레이어로 SOTA 달성
- 모델 경량화 및 학습 효율성 증대

## 코드 구현 학습 포인트

### 1. 텐서 차원 변환 이해
```python
# 핵심 차원 변환 과정
b,s,c -> b,c,s -> bc,n,w -> bc,n,d -> 1,bc,d -> predictions
```
- b: batch_size
- s: seq_len
- c: channels (enc_in)
- n: seg_num_x (입력 segment 개수)
- w: seg_len (segment 길이)
- d: d_model (hidden dimension)

### 2. Segment Embedding 구현
```python
self.valueEmbedding = nn.Sequential(
    nn.Linear(self.seg_len, self.d_model),  # W -> d
    nn.ReLU()
)
```
- Segment 내 정보를 압축하여 특징 추출
- Point들의 평균화 효과로 노이즈 감소

### 3. PMF의 위치/채널 임베딩
```python
# 위치 임베딩: 각 출력 segment의 시간적 위치
self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
# 채널 임베딩: 각 채널의 고유 특성
self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
```

### 4. RevIN 정규화 전략
- **Instance Normalization**: 각 시계열 인스턴스별 독립 정규화
- **Reversible**: 예측 후 원래 스케일 복원
- **Distribution Shift 해결**: 시간에 따른 분포 변화 대응

## 🔍 실험 및 분석 포인트

### 1. 하이퍼파라미터 선택
- **seg_len**: seq_len의 약수여야 함 (예: 96/48=2)
- **d_model**: Hidden dimension (논문은 512 사용)
- **dec_way**: PMF가 일반적으로 더 빠르고 성능 좋음
- **rnn_type**: GRU가 가장 효과적 (논문 기준)

### 2. 데이터셋별 특성
- **ETT 데이터**: 전기 변압기 온도 (시간별/15분별)
- **Traffic**: 도로 점유율 (outlier 많음 - RevIN 중요)
- **Weather**: 날씨 데이터 (21개 기상 지표)
- **Electricity**: 전력 소비량

### 3. 성능 평가 메트릭
- **MSE**: 큰 오차에 민감 (outlier 영향 큼)
- **MAE**: 절대 오차의 평균 (robust)

## 실습 체크리스트

### 기본 이해
- [ ] Segment 분할 과정 직접 그려보기
- [ ] 텐서 차원 변화 단계별 추적
- [ ] PMF vs RMF 차이점 코드로 확인

### 코드 분석
- [ ] VanillaRNN과 SegRNN 성능 비교 실험
- [ ] seg_len 변경에 따른 성능 변화 측정
- [ ] RevIN 유무에 따른 성능 차이 확인

### 심화 학습
- [ ] Ablation study 재현 (논문 Table 3)
- [ ] 다른 RNN 변형 (LSTM, vanilla RNN) 테스트
- [ ] 자체 데이터셋에 적용해보기

## 주요 수식 정리

### 1. Segmentation
```
X_seg = reshape(X, [B×C, N, W])
여기서 N = L/W (segment 개수)
```

### 2. Segment Embedding
```
H = ReLU(Linear(X_seg))
X_seg ∈ R^(BC×N×W) -> H ∈ R^(BC×N×D)
```

### 3. RNN Encoding
```
h_N = RNN(H)
최종 hidden state만 사용
```

### 4. PMF Decoding
```
Y = Linear(RNN(PE, h_N))
PE: Position Embedding
병렬 예측으로 속도 향상
```

## 구현 시 주의사항

1. **Segment 길이 선택**
   - seq_len의 약수여야 함
   - 너무 작으면 segment 수 증가 → 계산량 증가
   - 너무 크면 segment 내 정보 손실 가능

2. **메모리 관리**
   - Batch와 Channel을 합쳐서 처리 (BC 차원)
   - 대용량 데이터셋 처리 시 gradient accumulation 고려

3. **정규화 전략**
   - RevIN은 특히 분포가 변하는 데이터에 효과적
   - subtract_last vs subtract_mean 실험 필요

4. **학습 안정성**
   - Learning rate는 보수적으로 설정 (1e-4 추천)
   - Early stopping으로 과적합 방지

## 참고 자료

- **논문**: [SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting](https://arxiv.org/abs/2308.11200)
- **공식 저장소**: [https://github.com/lss-1138/SegRNN](https://github.com/lss-1138/SegRNN)
- **RevIN 논문**: [Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://openreview.net/pdf?id=cGDAkQo1C0p)

## 추가 탐구 아이디어

1. **Segment 길이 자동 선택**: 데이터 특성에 따른 최적 seg_len 탐색
2. **Adaptive Segmentation**: 고정 길이 대신 가변 segment
3. **Multi-scale Segmentation**: 여러 seg_len을 동시에 사용
4. **Cross-channel Attention**: 채널 간 관계 모델링 강화