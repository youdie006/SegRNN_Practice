import torch
import torch.nn as nn

class RevIN(nn.Module):
    """
    RevIN: Reversible Instance Normalization
    논문: "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift"
    
    핵심 아이디어:
    1. Instance Normalization: 각 시계열 인스턴스별로 독립적으로 정규화
    2. Reversible: 예측 후 원래 스케일로 복원 가능
    3. Distribution Shift 해결: 학습/테스트 데이터의 분포 차이 문제 완화
    
    시계열 예측에서 중요한 이유:
    - 시계열 데이터는 시간에 따라 평균과 분산이 변화 (non-stationary)
    - 정규화로 모델 학습 안정화, 역정규화로 원래 스케일 복원
    - 특히 다변량 시계열에서 각 변수의 스케일 차이 해결
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        Args:
            num_features: 채널(변수) 개수
            eps: 수치 안정성을 위한 작은 값
            affine: learnable affine 변환 사용 여부
            subtract_last: True면 마지막 값 빼기, False면 평균 빼기
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        """
        Args:
            x: 입력 텐서 (batch, seq_len, channels)
            mode: 'norm' (정규화) 또는 'denorm' (역정규화)
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: 
            raise NotImplementedError
        return x

    def _init_params(self):
        # Affine 변환 파라미터: 각 채널별 scale과 shift
        # 정규화 후 추가적인 변환으로 표현력 증가
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))   # scale
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))    # shift

    def _get_statistics(self, x):
        """
        각 인스턴스의 통계량 계산 및 저장
        - Instance-wise: 각 샘플, 각 채널별로 독립적으로 계산
        - 저장된 통계량은 denormalization에 사용
        """
        # dim2reduce: seq_len 차원 (중간 차원들)
        dim2reduce = tuple(range(1, x.ndim-1))
        
        if self.subtract_last:
            # 마지막 시점 값 저장 (단순하지만 효과적)
            # 시계열의 최근 레벨 정보 보존
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            # 시퀀스 평균 계산 (전통적 방법)
            # detach(): gradient 계산 제외 (통계량은 학습 대상 아님)
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        
        # 표준편차 계산 (모든 경우에 필요)
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        """
        정규화: (x - center) / scale
        - center: 평균 또는 마지막 값
        - scale: 표준편차
        """
        if self.subtract_last:
            x = x - self.last  # 마지막 값 중심화
        else:
            x = x - self.mean  # 평균 중심화
        
        x = x / self.stdev     # 스케일 정규화
        
        if self.affine:
            # Affine 변환으로 추가 조정
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        """
        역정규화: 원래 스케일로 복원
        정규화의 역순으로 진행
        """
        if self.affine:
            # Affine 역변환
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)  # 0 나누기 방지
        
        # 스케일 복원
        x = x * self.stdev
        
        # 중심 복원
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x