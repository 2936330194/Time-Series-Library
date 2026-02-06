"""
================================================================================
数据加载器模块 (Data Loader)
================================================================================

功能概述：
---------
本模块定义了多种 PyTorch Dataset 子类，用于加载和预处理不同类型的时序数据。
每个 Dataset 类负责：
    1. 读取原始数据文件（CSV/NPY等）
    2. 数据标准化（StandardScaler）
    3. 划分训练/验证/测试集
    4. 提取时间特征
    5. 生成滑动窗口样本

包含的数据集类：
---------------
【时序预测】
    - Dataset_ETT_hour:  ETT小时级数据（ETTh1, ETTh2）
    - Dataset_ETT_minute: ETT分钟级数据（ETTm1, ETTm2）
    - Dataset_Custom:    自定义CSV数据（用户自己的数据）
    - Dataset_M4:        M4竞赛数据

【异常检测】
    - PSMSegLoader, MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader

【时序分类】
    - UEAloader: UEA分类数据集

核心概念：
---------
序列长度参数 size = [seq_len, label_len, pred_len]

    |<------ seq_len ------>|
    |     encoder输入       |
    [x1, x2, x3, ..., x96]
                    |<- label_len ->|<-- pred_len -->|
                    |   decoder输入  |   预测目标      |
                    [x49, ..., x96, y1, y2, ..., y96]
                            |<----- 输出序列 (seq_y) ----->|

    - seq_len:   编码器输入序列长度（历史观测窗口）
    - label_len: 解码器起始token长度（从输入末尾截取，作为decoder的起始输入）
    - pred_len:  预测序列长度（未来预测窗口）

================================================================================
"""

import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
from datasets import load_dataset
from huggingface_hub import hf_hub_download
warnings.filterwarnings('ignore')

# HuggingFace 数据仓库地址（当本地数据不存在时，会从这里下载）
HUGGINGFACE_REPO = "thuml/Time-Series-Library"


# =============================================================================
#                         Dataset_ETT_hour 类
# =============================================================================
# 
# 用途：加载 ETT（Electricity Transformer Temperature）小时级数据集
# 数据来源：电力变压器的传感器数据，包含油温(OT)和6个负载特征
# 特点：数据集有固定的划分方式（12个月训练+4个月验证+4个月测试）
#
# =============================================================================

class Dataset_ETT_hour(Dataset):
    """
    ETT 小时级数据集加载器
    
    数据格式要求：
    -------------
    CSV文件，第一列为 'date'，其余列为特征，最后一列通常是目标变量 'OT'
    
    示例：
        date,HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
        2016-07-01 00:00:00,5.827,2.009,1.599,0.462,4.203,1.340,30.531
        2016-07-01 01:00:00,5.693,2.076,1.492,0.426,4.142,1.371,30.460
        ...
    
    参数说明：
    ---------
    args : Namespace
        配置参数对象
    root_path : str
        数据根目录，如 './dataset/'
    flag : str
        数据集划分标志，'train'/'val'/'test'
    size : list
        [seq_len, label_len, pred_len]，如 [96, 48, 96]
    features : str
        特征类型：
        - 'M': 多变量预测多变量 (Multivariate → Multivariate)
        - 'S': 单变量预测单变量 (Univariate → Univariate)
        - 'MS': 多变量预测单变量 (Multivariate → Univariate)
    data_path : str
        数据文件名，如 'ETTh1.csv'
    target : str
        目标变量列名，默认 'OT'（油温）
    scale : bool
        是否进行标准化，默认 True
    timeenc : int
        时间编码方式：0=传统编码，1=timeF编码
    freq : str
        时间频率，'h'=小时
    """
    
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        """
        初始化数据集
        """
        # =====================================================================
        # 1. 保存配置参数
        # =====================================================================
        self.args = args
        
        # 设置序列长度参数，如果未指定则使用默认值
        # 默认值：seq_len=384, label_len=96, pred_len=96
        if size == None:
            self.seq_len = 24 * 4 * 4    # 384 = 16天的小时数
            self.label_len = 24 * 4      # 96 = 4天
            self.pred_len = 24 * 4       # 96 = 4天
        else:
            self.seq_len = size[0]       # 编码器输入长度
            self.label_len = size[1]     # 解码器起始token长度
            self.pred_len = size[2]      # 预测长度
        
        # =====================================================================
        # 2. 确定数据集类型（训练/验证/测试）
        # =====================================================================
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]  # 0=训练, 1=验证, 2=测试

        # =====================================================================
        # 3. 保存其他配置
        # =====================================================================
        self.features = features    # 特征类型 'M'/'S'/'MS'
        self.target = target        # 目标列名
        self.scale = scale          # 是否标准化
        self.timeenc = timeenc      # 时间编码方式
        self.freq = freq            # 时间频率

        self.root_path = root_path  # 数据根目录
        self.data_path = data_path  # 数据文件名
        
        # =====================================================================
        # 4. 读取和处理数据
        # =====================================================================
        self.__read_data__()

    def __read_data__(self):
        """
        读取并预处理数据
        
        处理流程：
        1. 读取CSV文件
        2. 划分训练/验证/测试集边界
        3. 选择特征列
        4. 数据标准化（使用训练集的统计量）
        5. 提取时间特征
        """
        # 创建标准化器
        self.scaler = StandardScaler()

        # ---------------------------------------------------------------------
        # Step 1: 读取原始数据
        # ---------------------------------------------------------------------
        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        # 优先从本地读取，否则从 HuggingFace 下载
        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            df_raw = ds["train"].to_pandas()
        
        # ---------------------------------------------------------------------
        # Step 2: 定义数据集划分边界（ETT数据集的固定划分方式）
        # ---------------------------------------------------------------------
        # ETT数据集总共约20个月的数据
        # 划分方式：12个月训练 + 4个月验证 + 4个月测试
        # 
        # 边界计算：
        #   - 12个月 ≈ 12 * 30 * 24 = 8640 小时
        #   - 4个月  ≈ 4 * 30 * 24  = 2880 小时
        #
        # border1s: 每个数据集的起始索引（需要减去seq_len以确保有足够的历史数据）
        # border2s: 每个数据集的结束索引
        
        border1s = [0,                                                    # 训练集起始: 0
                    12 * 30 * 24 - self.seq_len,                          # 验证集起始: 8640 - seq_len
                    12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]            # 测试集起始: 11520 - seq_len
        border2s = [12 * 30 * 24,                                         # 训练集结束: 8640
                    12 * 30 * 24 + 4 * 30 * 24,                           # 验证集结束: 11520  
                    12 * 30 * 24 + 8 * 30 * 24]                           # 测试集结束: 14400
        
        # 根据 set_type 选择当前数据集的边界
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # ---------------------------------------------------------------------
        # Step 3: 选择特征列
        # ---------------------------------------------------------------------
        if self.features == 'M' or self.features == 'MS':
            # 多变量：使用除了date之外的所有列
            cols_data = df_raw.columns[1:]  # 跳过第一列'date'
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            # 单变量：只使用目标列
            df_data = df_raw[[self.target]]

        # ---------------------------------------------------------------------
        # Step 4: 数据标准化
        # ---------------------------------------------------------------------
        # 重要：使用训练集的统计量（均值和标准差）对所有数据进行标准化
        # 这样可以防止数据泄露
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]  # 只用训练集数据计算统计量
            self.scaler.fit(train_data.values)             # 拟合scaler
            data = self.scaler.transform(df_data.values)   # 转换所有数据
        else:
            data = df_data.values

        # ---------------------------------------------------------------------
        # Step 5: 提取时间特征
        # ---------------------------------------------------------------------
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            # 传统时间编码：提取月、日、星期、小时
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values  # [时间步数, 4]
        elif self.timeenc == 1:
            # timeF编码：使用傅里叶时间特征（更紧凑）
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)  # 转置为 [时间步数, 特征数]

        # ---------------------------------------------------------------------
        # Step 6: 保存处理后的数据
        # ---------------------------------------------------------------------
        self.data_x = data[border1:border2]  # 输入特征
        self.data_y = data[border1:border2]  # 输出目标（与输入相同，通过索引切片区分）

        # 数据增强（可选，仅在训练集上应用）
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp  # 时间特征

    def __getitem__(self, index):
        """
        获取单个样本
        
        滑动窗口采样示意图：
        ------------------------------------------------------------------
        原始数据:  [..., x_i, x_{i+1}, ..., x_{i+seq_len}, ..., x_{i+seq_len+pred_len}, ...]
                        |<-------- seq_x -------->|
                                    |<---- label_len ---->|<-- pred_len -->|
                                    |<----------- seq_y (label_len + pred_len) ---------->|
        ------------------------------------------------------------------
        
        参数：
        -----
        index : int
            样本索引（滑动窗口的起始位置）
            
        返回：
        -----
        seq_x : ndarray
            编码器输入序列 [seq_len, n_features]
        seq_y : ndarray  
            解码器输入+预测目标序列 [label_len + pred_len, n_features]
        seq_x_mark : ndarray
            编码器输入的时间特征 [seq_len, time_dim]
        seq_y_mark : ndarray
            解码器输入+预测的时间特征 [label_len + pred_len, time_dim]
        """
        # 计算各序列的起止索引
        s_begin = index                              # 输入序列起始
        s_end = s_begin + self.seq_len               # 输入序列结束
        r_begin = s_end - self.label_len             # 输出序列起始（从输入末尾回溯label_len）
        r_end = r_begin + self.label_len + self.pred_len  # 输出序列结束

        # 切片获取序列数据
        seq_x = self.data_x[s_begin:s_end]           # 编码器输入
        seq_y = self.data_y[r_begin:r_end]           # 解码器输入 + 预测目标
        seq_x_mark = self.data_stamp[s_begin:s_end]  # 输入时间特征
        seq_y_mark = self.data_stamp[r_begin:r_end]  # 输出时间特征

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """
        返回数据集中的样本总数
        
        计算公式：总长度 - seq_len - pred_len + 1
        （确保每个样本都有完整的输入和预测区间）
        """
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """
        将标准化后的数据转换回原始尺度
        
        用于：在预测完成后，将预测值还原为原始数值范围
        """
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            df_raw = ds["train"].to_pandas()

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# =============================================================================
#                         Dataset_Custom 类
# =============================================================================
#
# 用途：加载用户自定义的 CSV 数据集
# 特点：
#   1. 支持任意 CSV 格式的时序数据
#   2. 自动按照 70%/10%/20% 划分训练/验证/测试集
#   3. 最灵活的数据集类，推荐用于自己的数据
#
# 使用方法：
#   1. 准备 CSV 文件，格式要求：
#      - 第一列必须是 'date'（时间戳）
#      - 其余列为特征
#      - 目标变量可以放在任意位置（通过 target 参数指定）
#   2. 将 CSV 放到 root_path 目录下
#   3. 运行时指定 --data custom --data_path your_data.csv
#
# =============================================================================

class Dataset_Custom(Dataset):
    """
    自定义数据集加载器
    
    适用于：用户自己的 CSV 格式时序数据
    
    数据格式要求：
    -------------
    CSV文件，必须包含 'date' 列和目标变量列
    
    示例（电力负荷预测）：
        date,feature1,feature2,feature3,target
        2020-01-01 00:00:00,1.2,3.4,5.6,100.5
        2020-01-01 01:00:00,1.3,3.5,5.7,102.3
        ...
    
    示例（水库调度）：
        date,inflow,water_level,rainfall,outflow
        2020-01-01,150.2,98.5,0.0,145.0
        2020-01-02,160.5,99.1,5.2,155.0
        ...
    
    与 Dataset_ETT_hour 的区别：
    ---------------------------
    1. 数据划分：ETT使用固定边界，Custom按比例划分（70%/10%/20%）
    2. 列顺序：Custom 会自动将目标列移到最后
    3. 灵活性：Custom 适用于任意格式的 CSV
    
    参数说明：
    ---------
    args : Namespace
        配置参数对象
    root_path : str  
        数据根目录，如 './dataset/'
    flag : str
        数据集划分标志：'train'/'val'/'test'
    size : list
        [seq_len, label_len, pred_len]
    features : str
        特征类型：
        - 'M':  多变量预测多变量
        - 'S':  单变量预测单变量  
        - 'MS': 多变量预测单变量（最常用）
    data_path : str
        数据文件名，如 'my_data.csv'
    target : str
        目标变量列名（你想要预测的那一列）
    scale : bool
        是否标准化，默认 True
    timeenc : int
        时间编码方式：0=传统，1=timeF
    freq : str
        时间频率：'h'=小时, 't'=分钟, 'd'=天, 'w'=周, 'm'=月
    
    使用示例：
    ---------
    命令行：
        python run.py --data custom --data_path my_data.csv --target my_target \\
                      --seq_len 96 --pred_len 24 --features MS
    """
    
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        """
        初始化自定义数据集
        """
        # =====================================================================
        # 1. 设置序列长度参数
        # =====================================================================
        self.args = args
        
        if size == None:
            # 默认值（与ETT相同）
            self.seq_len = 24 * 4 * 4    # 384
            self.label_len = 24 * 4      # 96
            self.pred_len = 24 * 4       # 96
        else:
            self.seq_len = size[0]       # 编码器输入长度
            self.label_len = size[1]     # 解码器起始长度
            self.pred_len = size[2]      # 预测长度
        
        # =====================================================================
        # 2. 确定数据集类型
        # =====================================================================
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        # =====================================================================
        # 3. 保存配置
        # =====================================================================
        self.features = features    # 特征类型
        self.target = target        # 目标列名（你要预测的变量）
        self.scale = scale          # 是否标准化
        self.timeenc = timeenc      # 时间编码方式
        self.freq = freq            # 时间频率

        self.root_path = root_path  # 数据目录
        self.data_path = data_path  # 文件名
        
        # =====================================================================
        # 4. 读取数据
        # =====================================================================
        self.__read_data__()

    def __read_data__(self):
        """
        读取并预处理自定义数据
        
        关键步骤：
        1. 读取CSV
        2. 重新排列列顺序（date + 特征 + 目标）
        3. 按比例划分数据集（70%训练 / 10%验证 / 20%测试）
        4. 标准化
        5. 提取时间特征
        """
        self.scaler = StandardScaler()
        
        # ---------------------------------------------------------------------
        # Step 1: 读取原始数据
        # ---------------------------------------------------------------------
        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            # 如果本地没有，尝试从 HuggingFace 下载
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            split_name = "train" if "train" in ds else list(ds.keys())[0]
            df_raw = ds[split_name].to_pandas()

        # ---------------------------------------------------------------------
        # Step 2: 重新排列列顺序
        # ---------------------------------------------------------------------
        # 将目标列移到最后，确保格式统一：[date, 特征1, 特征2, ..., 目标]
        # 这是因为 features='MS' 模式下，输出只取最后一列
        cols = list(df_raw.columns)
        cols.remove(self.target)   # 移除目标列
        cols.remove('date')        # 移除日期列
        df_raw = df_raw[['date'] + cols + [self.target]]  # 重新排列
        
        # ---------------------------------------------------------------------
        # Step 3: 按比例划分数据集
        # ---------------------------------------------------------------------
        # 与 ETT 数据集的固定划分不同，Custom 使用百分比划分
        # 70% 训练 / 10% 验证 / 20% 测试
        num_train = int(len(df_raw) * 0.7)   # 训练集大小
        num_test = int(len(df_raw) * 0.2)    # 测试集大小
        num_vali = len(df_raw) - num_train - num_test  # 验证集大小
        
        # 计算各数据集的边界索引
        # 注意：起始索引需要减去 seq_len，确保有足够的历史数据
        border1s = [0,                                    # 训练集起始
                    num_train - self.seq_len,             # 验证集起始（需要回溯seq_len）
                    len(df_raw) - num_test - self.seq_len]  # 测试集起始
        border2s = [num_train,                            # 训练集结束
                    num_train + num_vali,                 # 验证集结束
                    len(df_raw)]                          # 测试集结束
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # ---------------------------------------------------------------------
        # Step 4: 选择特征列
        # ---------------------------------------------------------------------
        if self.features == 'M' or self.features == 'MS':
            # 多变量：使用所有特征列（不含date）
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            # 单变量：只使用目标列
            df_data = df_raw[[self.target]]

        # ---------------------------------------------------------------------
        # Step 5: 数据标准化
        # ---------------------------------------------------------------------
        # 重要：只用训练集数据拟合 scaler，防止数据泄露
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # ---------------------------------------------------------------------
        # Step 6: 提取时间特征
        # ---------------------------------------------------------------------
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        
        if self.timeenc == 0:
            # 传统编码：月、日、星期、小时
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            # timeF编码
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # ---------------------------------------------------------------------
        # Step 7: 保存处理后的数据
        # ---------------------------------------------------------------------
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        # 可选：数据增强
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        """
        获取单个样本（与 Dataset_ETT_hour 相同的逻辑）
        
        返回：
        -----
        seq_x : 编码器输入 [seq_len, n_features]
        seq_y : 解码器输入+目标 [label_len + pred_len, n_features]
        seq_x_mark : 输入时间特征
        seq_y_mark : 输出时间特征
        """
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        """返回样本总数"""
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        """将标准化数据转换回原始尺度"""
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        train_path = os.path.join(root_path, "train.csv")
        test_path = os.path.join(root_path, "test.csv")
        label_path = os.path.join(root_path, "test_label.csv")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_df      = pd.read_csv(train_path)
            test_df       = pd.read_csv(test_path)
            test_label_df = pd.read_csv(label_path)
        else:
            ds_data  = load_dataset(HUGGINGFACE_REPO, name="PSM-data")
            ds_label = load_dataset(HUGGINGFACE_REPO, name="PSM-label")
            train_df      = ds_data["train"].to_pandas()
            test_df       = ds_data["test"].to_pandas()
            test_label_df = ds_label[next(iter(ds_label))].to_pandas()

        data = train_df.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        test_data = test_df.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_label_df.values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "MSL_train.npy")
        test_path  = os.path.join(root_path, "MSL_test.npy")
        label_path = os.path.join(root_path, "MSL_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label  = np.load(label_path)

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data  = self.scaler.transform(test_data)

        self.train = train_data
        self.test  = test_data
        self.test_labels = test_label

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "SMAP_train.npy")
        test_path  = os.path.join(root_path, "SMAP_test.npy")
        label_path = os.path.join(root_path, "SMAP_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label = np.load(label_path)

        # 标准化
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data  = self.scaler.transform(test_data)

        self.train = train_data
        self.test  = test_data
        self.test_labels = test_label

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "SMD_train.npy")
        test_path  = os.path.join(root_path, "SMD_test.npy")
        label_path = os.path.join(root_path, "SMD_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label = np.load(label_path)
            
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_label
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train2_path = os.path.join(root_path, "swat_train2.csv")
        test_path   = os.path.join(root_path, "swat2.csv")
        if all(os.path.exists(p) for p in [train2_path, test_path]):
            train_data = pd.read_csv(train2_path)
            test_data   = pd.read_csv(test_path)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name="SWaT")
            train_data = ds["train"].to_pandas()
            test_data  = ds["test"].to_pandas()
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def _resolve_ts_path(self, root_path, dataset_name, flag):
        split = "TRAIN" if "train" in str(flag).lower() else "TEST"
        fname = f"{dataset_name}_{split}.ts"
        local = os.path.join(root_path, fname)
        if os.path.exists(local):
            return local
        return hf_hub_download(HUGGINGFACE_REPO, filename=f"{dataset_name}/{fname}", repo_type="dataset")

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        dataset_name = self.args.model_id
        ts_path = self._resolve_ts_path(root_path, dataset_name, flag or "train")

        all_df, labels_df = self.load_single(ts_path)
        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)
