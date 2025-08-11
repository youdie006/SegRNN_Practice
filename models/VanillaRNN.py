import torch
import torch.nn as nn
from layers.RevIN import RevIN


class Model(nn.Module):
    """
    VanillaRNN: 전통적인 Point-wise RNN 구현
    
    SegRNN과의 주요 차이점:
    1. Point-wise iteration: 각 시간 스텝을 개별적으로 처리 (SegRNN은 segment 단위)
    2. 직접 예측: 채널 값을 직접 예측 (SegRNN은 segment embedding 사용)
    3. 순차적 예측만 지원: autoregressive 방식만 가능 (SegRNN은 병렬 예측도 지원)
    
    문제점:
    - 메모리 복잡도: O(L) - 시퀀스 길이에 비례
    - 장기 의존성 학습 어려움: vanishing gradient 문제
    - 느린 예측 속도: pred_len만큼 순차적 반복 필요
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        # 기본 파라미터
        self.seq_len = configs.seq_len      # 입력 시퀀스 길이
        self.pred_len = configs.pred_len    # 예측 시퀀스 길이
        self.enc_in = configs.enc_in        # 입력 채널 수
        self.d_model = configs.d_model      # hidden dimension
        self.rnn_type = configs.rnn_type

        # RNN 레이어
        # 주목: 입력 차원이 enc_in (SegRNN은 d_model 사용)
        # 각 시간 스텝의 모든 채널을 직접 입력으로 받음
        assert self.rnn_type in ['rnn', 'gru', 'lstm']
        if self.rnn_type == "rnn":
            self.rnn = nn.RNN(input_size=self.enc_in, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.enc_in, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)
        elif self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.enc_in, hidden_size=self.d_model, num_layers=1, bias=True,
                              batch_first=True, bidirectional=False)

        # 예측 레이어: hidden state -> 다음 시간 스텝의 모든 채널 값
        self.predict = nn.Sequential(
            nn.Linear(self.d_model, self.enc_in)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        x = x_enc  # shape: (b, seq_len, enc_in)

        # Step 1: Encoding
        # 전체 입력 시퀀스를 처리하여 최종 hidden state 획득
        # 각 시간 스텝마다 RNN 업데이트 (총 seq_len번)
        if self.rnn_type == "lstm":
            _, (hn, cn) = self.rnn(x)
        else:
            _, hn = self.rnn(x)  # hn shape: (1, b, d_model)

        # Step 2: Decoding (Autoregressive 예측)
        # pred_len만큼 반복하며 한 스텝씩 예측
        # SegRNN과 달리 segment 단위가 아닌 point 단위 예측
        y = []
        if self.rnn_type == "lstm":
            for i in range(self.pred_len):
                # hidden state로부터 다음 시간 스텝 예측
                yy = self.predict(hn)        # (1, b, enc_in)
                yy = yy.permute(1, 0, 2)     # (b, 1, enc_in)
                y.append(yy)
                # 예측값을 다시 입력으로 사용 (autoregressive)
                _, (hn, cn) = self.rnn(yy, (hn, cn))
        else:
            for i in range(self.pred_len):
                yy = self.predict(hn)         # (1, b, enc_in)
                yy = yy.permute(1,0,2)        # (b, 1, enc_in)
                y.append(yy)
                _, hn = self.rnn(yy, hn)
        
        # 모든 예측 결합: (b, pred_len, enc_in)
        y = torch.stack(y, dim=1).squeeze(2)

        return y

