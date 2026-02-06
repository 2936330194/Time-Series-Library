"""
Time-Series-Library 快速运行配置脚本
=====================================
使用方法：直接运行 python my_run.py
修改下方的配置参数即可自定义实验

作者：根据 TSLib 项目创建
"""

import argparse
import os
import torch
import random
import numpy as np

# ============================================================================
#                           🎯 快速配置区（主要修改这里）
# ============================================================================

# ---------------------- 任务配置 ----------------------
TASK_NAME = 'long_term_forecast'  
# 可选任务：
#   - 'long_term_forecast'  : 长期预测
#   - 'short_term_forecast' : 短期预测
#   - 'imputation'          : 缺失值填充
#   - 'anomaly_detection'   : 异常检测
#   - 'classification'      : 分类

IS_TRAINING = 1  # 1=训练+测试, 0=仅测试（需要已有checkpoint）

# ---------------------- 模型配置 ----------------------
MODEL = 'DLinear'
# 推荐模型（按复杂度排序）：
#   简单快速: 'DLinear', 'NLinear', 'Linear'
#   中等性能: 'PatchTST', 'TimeMixer', 'iTransformer'
#   较重但强: 'TimesNet', 'Autoformer', 'FEDformer'
#   最新SOTA: 'TimeXer' (需要外生变量)

MODEL_ID = 'ETTh1_96_96'  # 实验标识，格式建议：数据集_输入长度_预测长度

# ---------------------- 数据配置 ----------------------
DATA = 'ETTh1'                    # 数据集类型
ROOT_PATH = './dataset/'          # 数据根目录
DATA_PATH = 'ETTh1.csv'           # 数据文件名
# 可用数据集：
#   ETT系列: ETTh1.csv, ETTh2.csv, ETTm1.csv, ETTm2.csv
#   其他: electricity.csv, weather.csv, traffic.csv, exchange_rate.csv

FEATURES = 'M'
# 预测类型：
#   'M'  : 多变量预测多变量 (Multivariate -> Multivariate)
#   'S'  : 单变量预测单变量 (Univariate -> Univariate)
#   'MS' : 多变量预测单变量 (Multivariate -> Single target)

TARGET = 'OT'  # 目标变量名（用于 S 或 MS 任务）

# ---------------------- 序列长度配置 ----------------------
SEQ_LEN = 96      # 输入序列长度（历史窗口）
LABEL_LEN = 48    # 标签长度（decoder的起始token）
PRED_LEN = 96     # 预测长度

# ---------------------- 模型结构配置 ----------------------
ENC_IN = 7        # 编码器输入特征数（ETTh1有7个特征）
DEC_IN = 7        # 解码器输入特征数
C_OUT = 7         # 输出特征数
D_MODEL = 512     # 模型维度
N_HEADS = 8       # 注意力头数
E_LAYERS = 2      # 编码器层数
D_LAYERS = 1      # 解码器层数
D_FF = 2048       # 前馈网络维度
DROPOUT = 0.1     # Dropout率

# ---------------------- 训练配置 ----------------------
TRAIN_EPOCHS = 10    # 训练轮数
BATCH_SIZE = 32      # 批次大小
LEARNING_RATE = 0.0001  # 学习率
PATIENCE = 3         # 早停耐心值
NUM_WORKERS = 0      # 数据加载线程数（Windows建议设为0）

# ---------------------- GPU配置 ----------------------
USE_GPU = True       # 是否使用GPU
GPU = 0              # GPU编号

# ---------------------- 其他配置 ----------------------
DES = 'Exp'          # 实验描述
ITR = 1              # 重复实验次数


# ============================================================================
#                           🔧 以下代码无需修改
# ============================================================================

def main():
    # 设置随机种子
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    
    # 构建参数
    args = argparse.Namespace(
        # 基本配置
        task_name=TASK_NAME,
        is_training=IS_TRAINING,
        model_id=MODEL_ID,
        model=MODEL,
        
        # 数据配置
        data=DATA,
        root_path=ROOT_PATH,
        data_path=DATA_PATH,
        features=FEATURES,
        target=TARGET,
        freq='h',
        checkpoints='./checkpoints/',
        
        # 序列长度
        seq_len=SEQ_LEN,
        label_len=LABEL_LEN,
        pred_len=PRED_LEN,
        seasonal_patterns='Monthly',
        inverse=False,
        
        # 模型结构
        enc_in=ENC_IN,
        dec_in=DEC_IN,
        c_out=C_OUT,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        e_layers=E_LAYERS,
        d_layers=D_LAYERS,
        d_ff=D_FF,
        dropout=DROPOUT,
        embed='timeF',
        activation='gelu',
        factor=1,
        distil=True,
        
        # 训练配置
        num_workers=NUM_WORKERS,
        itr=ITR,
        train_epochs=TRAIN_EPOCHS,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        learning_rate=LEARNING_RATE,
        des=DES,
        loss='MSE',
        lradj='type1',
        use_amp=False,
        
        # GPU配置
        use_gpu=USE_GPU,
        gpu=GPU,
        gpu_type='cuda',
        use_multi_gpu=False,
        devices='0,1,2,3',
        
        # 其他模型特定参数
        top_k=5,
        num_kernels=6,
        expand=2,
        d_conv=4,
        moving_avg=25,
        channel_independence=1,
        decomp_method='moving_avg',
        use_norm=1,
        down_sampling_layers=0,
        down_sampling_window=1,
        down_sampling_method=None,
        seg_len=96,
        
        # 任务特定参数
        mask_rate=0.25,
        anomaly_ratio=0.25,
        
        # 投影器参数
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        
        # DTW指标
        use_dtw=False,
        
        # 数据增强
        augmentation_ratio=0,
        seed=2,
        jitter=False,
        scaling=False,
        permutation=False,
        randompermutation=False,
        magwarp=False,
        timewarp=False,
        windowslice=False,
        windowwarp=False,
        rotation=False,
        spawner=False,
        dtwwarp=False,
        shapedtwwarp=False,
        wdba=False,
        discdtw=False,
        discsdtw=False,
        extra_tag='',
        
        # TimeXer
        patch_len=16,
        
        # GCN参数
        node_dim=10,
        gcn_depth=2,
        gcn_dropout=0.3,
        propalpha=0.3,
        conv_channel=32,
        skip_channel=32,
        
        # DLinear
        individual=False,
        
        # TimeFilter
        alpha=0.1,
        top_p=0.5,
        pos=1,
    )
    
    # 设置设备
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('✅ 使用 GPU:', torch.cuda.get_device_name(args.gpu))
    else:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
            print('✅ 使用 Apple MPS')
        else:
            args.device = torch.device("cpu")
            print('⚠️ 使用 CPU（训练会较慢）')
    
    # 打印配置摘要
    print('\n' + '='*60)
    print('📊 Time-Series-Library 实验配置')
    print('='*60)
    print(f'  任务类型: {args.task_name}')
    print(f'  模型: {args.model}')
    print(f'  数据集: {args.data} ({args.data_path})')
    print(f'  序列配置: 输入{args.seq_len} → 预测{args.pred_len}')
    print(f'  训练轮数: {args.train_epochs}')
    print(f'  批次大小: {args.batch_size}')
    print('='*60 + '\n')
    
    # 导入对应的实验类
    if args.task_name == 'long_term_forecast':
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'short_term_forecast':
        from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
        Exp = Exp_Short_Term_Forecast
    elif args.task_name == 'imputation':
        from exp.exp_imputation import Exp_Imputation
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        from exp.exp_anomaly_detection import Exp_Anomaly_Detection
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'classification':
        from exp.exp_classification import Exp_Classification
        Exp = Exp_Classification
    else:
        from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
        Exp = Exp_Long_Term_Forecast
    
    # 运行实验
    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_name, args.model_id, args.model, args.data,
                args.features, args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
                args.expand, args.d_conv, args.factor, args.embed, args.distil,
                args.des, ii)
            
            print(f'>>> 开始训练: {setting}')
            exp.train(setting)
            
            print(f'>>> 开始测试: {setting}')
            exp.test(setting)
            
            # 清理GPU缓存
            if args.use_gpu and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        print('\n✅ 实验完成！')
        print(f'📁 模型保存在: ./checkpoints/')
        print(f'📁 结果保存在: ./results/')
    else:
        exp = Exp(args)
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.task_name, args.model_id, args.model, args.data,
            args.features, args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
            args.expand, args.d_conv, args.factor, args.embed, args.distil,
            args.des, 0)
        
        print(f'>>> 仅测试模式: {setting}')
        exp.test(setting, test=1)
        
        if args.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
