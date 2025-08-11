import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

parser = argparse.ArgumentParser(description='Model family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2024, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task (예측 작업 설정)
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')      # 입력 시퀀스 길이 (lookback window)
parser.add_argument('--label_len', type=int, default=0, help='start token length')         # Transformer용 시작 토큰 길이 (SegRNN은 사용 안함)
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length') # 예측 시퀀스 길이 (forecast horizon)

# SegRNN 특화 파라미터
parser.add_argument('--rnn_type', default='gru', help='RNN 종류: rnn/gru/lstm (논문은 GRU 사용)')  
parser.add_argument('--dec_way', default='pmf', help='디코딩 방식: pmf(병렬)/rmf(순차) - PMF가 더 빠름')
parser.add_argument('--seg_len', type=int, default=48, help='segment 길이 - seq_len의 약수여야 함')
parser.add_argument('--channel_id', type=int, default=1, help='채널별 위치 인코딩 사용 여부 (1: 사용)')

# DLinear
#parser.add_argument('--individual', action='store_true', default=False, help='DLinear: a linear layer for each variate(channel) individually')

# PatchTST (패치 기반 Transformer)
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='패치 길이 (SegRNN의 seg_len과 유사한 개념)')
parser.add_argument('--stride', type=int, default=8, help='패치 간 stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN 정규화 사용 (1: 사용, 0: 미사용)')
parser.add_argument('--affine', type=int, default=0, help='RevIN affine 변환 사용')
parser.add_argument('--subtract_last', type=int, default=0, help='0: 평균 빼기, 1: 마지막 값 빼기')
parser.add_argument('--decomposition', type=int, default=0, help='시계열 분해 사용 여부')
parser.add_argument('--kernel_size', type=int, default=25, help='분해용 커널 크기')
parser.add_argument('--individual', type=int, default=0, help='채널별 개별 헤드 사용')

# Transformer 계열 모델 공통 파라미터
parser.add_argument('--embed_type', type=int, default=0, help='임베딩 타입 선택')
parser.add_argument('--enc_in', type=int, default=7, help='입력 채널/변수 개수')  # SegRNN에서도 사용
parser.add_argument('--dec_in', type=int, default=7, help='디코더 입력 크기')
parser.add_argument('--c_out', type=int, default=7, help='출력 채널 수')
parser.add_argument('--d_model', type=int, default=512, help='모델 hidden dimension (SegRNN도 사용)')
parser.add_argument('--n_heads', type=int, default=8, help='attention head 개수')
parser.add_argument('--e_layers', type=int, default=2, help='인코더 레이어 수')
parser.add_argument('--d_layers', type=int, default=1, help='디코더 레이어 수')
parser.add_argument('--d_ff', type=int, default=2048, help='FFN hidden dimension')
parser.add_argument('--moving_avg', type=int, default=25, help='이동평균 윈도우 크기')
parser.add_argument('--factor', type=int, default=1, help='attention factor (Informer)')
parser.add_argument('--distil', action='store_false',
                    help='Informer distilling 사용 여부',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.5, help='dropout 비율 (SegRNN도 사용)')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# 학습 최적화 설정
parser.add_argument('--num_workers', type=int, default=10, help='데이터 로더 워커 수')
parser.add_argument('--itr', type=int, default=2, help='실험 반복 횟수 (여러 시드로 실험)')
parser.add_argument('--train_epochs', type=int, default=30, help='학습 에폭 수')
parser.add_argument('--batch_size', type=int, default=128, help='배치 크기')
parser.add_argument('--patience', type=int, default=5, help='early stopping patience (성능 개선 없을 시 대기 에폭)')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='학습률')
parser.add_argument('--des', type=str, default='test', help='실험 설명 (결과 저장시 사용)')
parser.add_argument('--loss', type=str, default='mse', help='손실 함수: mse/mae')
parser.add_argument('--lradj', type=str, default='type3', help='학습률 스케줄러 타입')
parser.add_argument('--pct_start', type=float, default=0.3, help='OneCycleLR warm-up 비율')
parser.add_argument('--use_amp', action='store_true', help='AMP(Automatic Mixed Precision) 사용', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # 실험 설정 문자열 생성 (결과 저장 경로에 사용)
        # SegRNN 특화 파라미터 포함: rnn_type, dec_way, seg_len
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_dm{}_dr{}_rt{}_dw{}_sl{}_{}_{}_{}'.format(
            args.model_id,      # 실험 ID
            args.model,         # 모델명 (SegRNN 등)
            args.data,          # 데이터셋명
            args.features,      # M/S/MS
            args.seq_len,       # 입력 길이
            args.pred_len,      # 예측 길이
            args.d_model,       # hidden dimension
            args.dropout,       # dropout 비율
            args.rnn_type,      # RNN 종류 (SegRNN용)
            args.dec_way,       # 디코딩 방식 (SegRNN용)
            args.seg_len,       # segment 길이 (SegRNN용)
            args.loss,          # 손실 함수
            args.des,           # 실험 설명
            ii)                 # 반복 번호

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_dm{}_dr{}_rt{}_dw{}_sl{}_{}_{}_{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.d_model,
        args.dropout,
        args.rnn_type,
        args.dec_way,
        args.seg_len,
        args.loss,
        args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
