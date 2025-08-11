'''
SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting
논문: https://arxiv.org/abs/2308.11200

핵심 혁신:
1. Segment-wise iteration: point-wise 대신 segment 단위로 처리하여 메모리 효율성과 장기 의존성 학습 개선
2. PMF (Parallel Multi-step Forecasting): 병렬 예측으로 속도 향상
3. 단일 GRU 레이어로 SOTA 성능 달성
'''

import torch
import torch.nn as nn
from layers.RevIN import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # 시계열 길이 파라미터
        self.seq_len = configs.seq_len      # 입력 시퀀스 길이 (lookback window)
        self.pred_len = configs.pred_len    # 예측 시퀀스 길이 (forecast horizon)
        self.enc_in = configs.enc_in        # 입력 채널 수 (변수 개수)
        self.d_model = configs.d_model      # 모델 hidden dimension
        self.dropout = configs.dropout

        # SegRNN 특화 파라미터
        self.rnn_type = configs.rnn_type    # RNN 종류: 'rnn', 'gru', 'lstm'
        self.dec_way = configs.dec_way      # 디코딩 방식: 'rmf'(순차적), 'pmf'(병렬)
        self.seg_len = configs.seg_len      # segment 길이 (논문에서 W로 표기)
        self.channel_id = configs.channel_id # 채널별 위치 인코딩 사용 여부
        self.revin = configs.revin          # RevIN 정규화 사용 여부

        assert self.rnn_type in ['rnn', 'gru', 'lstm']
        assert self.dec_way in ['rmf', 'pmf']

        # segment 개수 계산
        # 예: seq_len=96, seg_len=48 -> seg_num_x=2 (입력을 2개 segment로 분할)
        self.seg_num_x = self.seq_len//self.seg_len

        # Segment Embedding Layer
        # segment 내 포인트들을 d_model 차원으로 매핑
        # 이를 통해 segment 내 정보를 압축하고 특징 추출
        self.valueEmbedding = nn.Sequential(
            nn.Linear(self.seg_len, self.d_model),  # W -> d
            nn.ReLU()
        )

        # RNN 레이어 선택 (단일 레이어만 사용 - 논문의 핵심 포인트)
        # SegRNN은 단 하나의 RNN 레이어로 SOTA 성능 달성
        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)

        # 디코딩 전략 설정
        if self.dec_way == "rmf":  # Recurrent Multi-step Forecasting
            # RMF: 순차적으로 다음 segment 예측 (autoregressive)
            self.seg_num_y = self.pred_len // self.seg_len  # 출력 segment 개수
            self.predict = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.seg_len)  # hidden state -> segment
            )
        elif self.dec_way == "pmf":  # Parallel Multi-step Forecasting
            # PMF: 모든 미래 segment를 병렬로 예측 (non-autoregressive)
            self.seg_num_y = self.pred_len // self.seg_len
            
            if self.channel_id:
                # 위치 임베딩과 채널 임베딩을 분리하여 학습
                # 각 segment의 시간적 위치와 각 채널의 특성을 독립적으로 모델링
                self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))      # 시간 위치 정보
                self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))     # 채널별 특성
            else:
                # 위치 임베딩만 사용 (모든 채널 동일)
                self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model))

            self.predict = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.seg_len)
            )
        
        # RevIN (Reversible Instance Normalization)
        # 시계열의 분포 변화(distribution shift) 문제 해결
        if self.revin:
            self.revinLayer = RevIN(self.enc_in, affine=False, subtract_last=False)


    def forward(self, x):
        '''
        입력: x shape = (batch_size, seq_len, enc_in)
        출력: y shape = (batch_size, pred_len, enc_in)
        
        텐서 차원 표기:
        b: batch_size, c: channel_size (enc_in), s: seq_len
        d: d_model, w: seg_len, n: seg_num_x, m: seg_num_y
        '''
        batch_size = x.size(0)

        # Step 1: 정규화 및 차원 변환
        # b,s,c -> b,c,s (채널을 별도 차원으로 분리하여 처리)
        if self.revin:
            # RevIN으로 instance-wise 정규화
            x = self.revinLayer(x, 'norm').permute(0, 2, 1)
        else:
            # 마지막 값 빼기 (simple normalization)
            seq_last = x[:, -1:, :].detach()
            x = (x - seq_last).permute(0, 2, 1)  # b,c,s

        # Step 2: Segmentation 및 Embedding
        # b,c,s -> bc,n,w (reshape: 채널별로 segment 분할)
        # bc,n,w -> bc,n,d (embedding: 각 segment를 d_model 차원으로 매핑)
        # 이 과정이 SegRNN의 핵심: point-wise가 아닌 segment-wise 처리
        x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))

        # Step 3: RNN Encoding
        # segment 시퀀스를 RNN으로 인코딩하여 최종 hidden state 획득
        # 입력: (bc, n, d) - bc개 시퀀스, 각각 n개 segment
        # 출력: hn (1, bc, d) - 마지막 hidden state
        if self.rnn_type == "lstm":
            _, (hn, cn) = self.rnn(x)
        else:
            _, hn = self.rnn(x)  # bc,n,d -> 1,bc,d

        # Step 4: Decoding (예측 생성)
        if self.dec_way == "rmf":  # Recurrent Multi-step Forecasting
            # RMF: 순차적으로 각 미래 segment 예측
            # 이전 예측을 다음 예측의 입력으로 사용 (autoregressive)
            y = []
            for i in range(self.seg_num_y):
                # hidden state로부터 다음 segment 예측
                yy = self.predict(hn)      # 1,bc,w (w=seg_len)
                yy = yy.permute(1,0,2)     # bc,1,w
                y.append(yy)
                
                # 예측된 segment를 다시 임베딩하여 다음 스텝 입력으로 사용
                yy = self.valueEmbedding(yy)  # bc,1,w -> bc,1,d
                if self.rnn_type == "lstm":
                    _, (hn, cn) = self.rnn(yy, (hn, cn))
                else:
                    _, hn = self.rnn(yy, hn)
            
            # 모든 예측 segment 결합: [bc,1,w] * m -> b,c,pred_len
            y = torch.stack(y, dim=1).squeeze(2).reshape(-1, self.enc_in, self.pred_len)
            
        elif self.dec_way == "pmf":  # Parallel Multi-step Forecasting
            # PMF: 모든 미래 segment를 병렬로 예측 (non-autoregressive)
            # 위치 임베딩을 사용하여 각 segment의 시간적 위치 구분
            
            if self.channel_id:
                # 위치 임베딩과 채널 임베딩 결합
                # pos_emb: (m, d//2) -> (c, m, d//2)
                # channel_emb: (c, d//2) -> (c, m, d//2)
                # 최종: (bcm, 1, d)
                pos_emb = torch.cat([
                    self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),      # 시간 위치
                    self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1) # 채널 정보
                ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size,1,1)
            else:
                # 위치 임베딩만 사용: (m, d) -> (bcm, 1, d)
                pos_emb = self.pos_emb.repeat(batch_size * self.enc_in, 1).unsqueeze(1)

            # RNN에 위치 임베딩 입력, hidden state는 모든 출력 위치로 복제
            # hn: (1,bc,d) -> (1,bcm,d) (각 출력 segment마다 복사)
            if self.rnn_type == "lstm":
                _, (hy, cy) = self.rnn(pos_emb,
                                       (hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model),
                                        cn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)))
            else:
                _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model))
            
            # 최종 예측: (1,bcm,d) -> (1,bcm,w) -> (b,c,pred_len)
            y = self.predict(hy).view(-1, self.enc_in, self.pred_len)

        # Step 5: 역정규화 및 차원 복원
        if self.revin:
            # RevIN 역변환: 원래 스케일로 복원
            y = self.revinLayer(y.permute(0, 2, 1), 'denorm')
        else:
            # 마지막 값 더하기 (simple denormalization)
            y = y.permute(0, 2, 1) + seq_last

        return y

'''
Concise version implementation that only includes necessary code
'''
# import torch
# import torch.nn as nn
#
# class Model(nn.Module):
#     def __init__(self, configs):
#         super(Model, self).__init__()
#
#         # get parameters
#         self.seq_len = configs.seq_len
#         self.pred_len = configs.pred_len
#         self.enc_in = configs.enc_in
#         self.d_model = configs.d_model
#         self.dropout = configs.dropout
#
#         self.seg_len = configs.seg_len
#         self.seg_num_x = self.seq_len//self.seg_len
#         self.seg_num_y = self.pred_len // self.seg_len
#
#
#         self.valueEmbedding = nn.Sequential(
#             nn.Linear(self.seg_len, self.d_model),
#             nn.ReLU()
#         )
#         self.rnn = nn.GRU(input_size=self.d_model, hidden_size=self.d_model, num_layers=1, bias=True,
#                               batch_first=True, bidirectional=False)
#         self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, self.d_model // 2))
#         self.channel_emb = nn.Parameter(torch.randn(self.enc_in, self.d_model // 2))
#         self.predict = nn.Sequential(
#             nn.Dropout(self.dropout),
#             nn.Linear(self.d_model, self.seg_len)
#         )
#
#     def forward(self, x):
#         # b:batch_size c:channel_size s:seq_len s:seq_len
#         # d:d_model w:seg_len n:seg_num_x m:seg_num_y
#         batch_size = x.size(0)
#
#         # normalization and permute     b,s,c -> b,c,s
#         seq_last = x[:, -1:, :].detach()
#         x = (x - seq_last).permute(0, 2, 1) # b,c,s
#
#         # segment and embedding    b,c,s -> bc,n,w -> bc,n,d
#         x = self.valueEmbedding(x.reshape(-1, self.seg_num_x, self.seg_len))
#
#         # encoding
#         _, hn = self.rnn(x) # bc,n,d  1,bc,d
#
#         # m,d//2 -> 1,m,d//2 -> c,m,d//2
#         # c,d//2 -> c,1,d//2 -> c,m,d//2
#         # c,m,d -> cm,1,d -> bcm, 1, d
#         pos_emb = torch.cat([
#             self.pos_emb.unsqueeze(0).repeat(self.enc_in, 1, 1),
#             self.channel_emb.unsqueeze(1).repeat(1, self.seg_num_y, 1)
#         ], dim=-1).view(-1, 1, self.d_model).repeat(batch_size,1,1)
#
#         _, hy = self.rnn(pos_emb, hn.repeat(1, 1, self.seg_num_y).view(1, -1, self.d_model)) # bcm,1,d  1,bcm,d
#
#         # 1,bcm,d -> 1,bcm,w -> b,c,s
#         y = self.predict(hy).view(-1, self.enc_in, self.pred_len)
#
#         # permute and denorm
#         y = y.permute(0, 2, 1) + seq_last
#
#         return y