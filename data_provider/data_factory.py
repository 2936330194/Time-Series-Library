"""
================================================================================
数据工厂模块 (Data Factory)
================================================================================

功能概述：
---------
这是 Time-Series-Library 项目的数据提供核心模块，采用"工厂模式"设计。
它的主要职责是：
    1. 根据用户配置，选择合适的数据集类（Dataset Class）
    2. 创建数据集对象（Dataset）和数据加载器（DataLoader）
    3. 支持不同任务类型（预测、异常检测、分类）的数据处理

工作流程：
---------
    用户配置 (args) 
         ↓
    data_provider() 函数
         ↓
    根据 args.data 选择数据集类
         ↓
    根据 args.task_name 选择初始化方式
         ↓
    返回 (Dataset, DataLoader)

使用示例：
---------
    # 在训练/验证/测试阶段调用
    train_data, train_loader = data_provider(args, flag='train')
    val_data, val_loader = data_provider(args, flag='val')
    test_data, test_loader = data_provider(args, flag='test')

================================================================================
"""

# =============================================================================
#                              导入依赖模块
# =============================================================================

# 从 data_loader.py 导入各种数据集类
# 每个数据集类都是 torch.utils.data.Dataset 的子类，负责特定格式数据的加载和预处理
from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader

# 导入 UEA 数据集的自定义 collate 函数（用于分类任务中处理变长序列）
from data_provider.uea import collate_fn

# PyTorch 的 DataLoader，用于批量加载数据、打乱顺序、多进程加速等
from torch.utils.data import DataLoader


# =============================================================================
#                         数据集字典 (Dataset Registry)
# =============================================================================
# 
# 这是一个"注册表"，将数据集名称字符串映射到对应的数据集类
# 当用户通过 --data 参数指定数据集时，会从这里查找对应的类
#
# 数据集类型说明：
# ---------------
# 【时序预测数据集】
#   - ETTh1, ETTh2: 电力变压器温度数据（小时级），常用于 benchmark
#   - ETTm1, ETTm2: 电力变压器温度数据（分钟级）
#   - custom: 自定义数据集，用于加载用户自己的 CSV 数据
#   - m4: M4 竞赛数据集，用于短期预测任务
#
# 【异常检测数据集】
#   - PSM: 服务器机器数据集 (Server Machine Dataset from eBay)
#   - MSL: 火星科学实验室数据集 (Mars Science Laboratory)
#   - SMAP: 土壤湿度主动/被动卫星数据 (Soil Moisture Active Passive)
#   - SMD: 服务器机器数据集 (Server Machine Dataset)
#   - SWAT: 安全水处理数据集 (Secure Water Treatment)
#
# 【分类数据集】
#   - UEA: UEA 时序分类数据集合集

data_dict = {
    # ===== 时序预测数据集 =====
    'ETTh1': Dataset_ETT_hour,      # 电力变压器温度 - 小时级 - 数据集1
    'ETTh2': Dataset_ETT_hour,      # 电力变压器温度 - 小时级 - 数据集2
    'ETTm1': Dataset_ETT_minute,    # 电力变压器温度 - 分钟级 - 数据集1
    'ETTm2': Dataset_ETT_minute,    # 电力变压器温度 - 分钟级 - 数据集2
    'custom': Dataset_Custom,       # 自定义数据集（用户自己的CSV文件）
    'm4': Dataset_M4,               # M4 竞赛数据集（短期预测）
    
    # ===== 异常检测数据集 =====
    'PSM': PSMSegLoader,            # eBay 服务器机器数据
    'MSL': MSLSegLoader,            # NASA 火星实验室数据
    'SMAP': SMAPSegLoader,          # NASA 卫星土壤湿度数据
    'SMD': SMDSegLoader,            # 服务器机器数据
    'SWAT': SWATSegLoader,          # 水处理系统安全数据
    
    # ===== 分类数据集 =====
    'UEA': UEAloader                # UEA 时序分类通用加载器
}


# =============================================================================
#                         数据提供函数 (Data Provider)
# =============================================================================

def data_provider(args, flag):
    """
    数据提供函数 —— 整个数据加载流程的入口
    
    功能：
    -----
    根据配置参数(args)和数据阶段(flag)，创建并返回数据集和数据加载器
    
    参数：
    -----
    args : Namespace
        包含所有配置参数的命名空间对象，来自命令行解析
        重要参数包括:
        - args.data: 数据集名称 ('ETTh1', 'custom', 等)
        - args.task_name: 任务类型 ('long_term_forecast', 'anomaly_detection', 等)
        - args.batch_size: 批次大小
        - args.seq_len: 输入序列长度
        - args.pred_len: 预测序列长度
        - args.label_len: 标签序列长度（用于decoder）
        
    flag : str
        数据集划分标志，可选值:
        - 'train': 训练集
        - 'val': 验证集
        - 'test': 测试集
    
    返回：
    -----
    data_set : Dataset
        PyTorch Dataset 对象，可通过索引访问单个样本
        
    data_loader : DataLoader
        PyTorch DataLoader 对象，用于批量迭代数据
    
    使用示例：
    ---------
    >>> train_data, train_loader = data_provider(args, flag='train')
    >>> for batch_x, batch_y, batch_x_mark, batch_y_mark in train_loader:
    ...     # batch_x: 输入序列 [batch_size, seq_len, features]
    ...     # batch_y: 输出序列 [batch_size, label_len + pred_len, features]
    ...     # batch_x_mark: 输入时间特征 [batch_size, seq_len, time_features]
    ...     # batch_y_mark: 输出时间特征 [batch_size, label_len + pred_len, time_features]
    ...     pass
    """
    
    # -------------------------------------------------------------------------
    # Step 1: 根据 args.data 从注册表中获取对应的数据集类
    # -------------------------------------------------------------------------
    # 例如: args.data = 'ETTh1' → Data = Dataset_ETT_hour
    Data = data_dict[args.data]
    
    # -------------------------------------------------------------------------
    # Step 2: 确定时间编码方式
    # -------------------------------------------------------------------------
    # timeenc 控制时间特征的编码方式:
    #   - 0: 使用传统时间特征 (月、日、周几、小时等，稀疏编码)
    #   - 1: 使用 timeF 编码 (更紧凑的傅里叶时间特征)
    timeenc = 0 if args.embed != 'timeF' else 1

    # -------------------------------------------------------------------------
    # Step 3: 设置 DataLoader 的通用参数
    # -------------------------------------------------------------------------
    # 测试集不打乱顺序，训练集和验证集需要打乱
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    
    # drop_last: 是否丢弃最后一个不完整的 batch
    # 默认为 False，保留所有数据
    drop_last = False
    
    # 批次大小
    batch_size = args.batch_size
    
    # 数据频率 (如 'h'=小时, 't'=分钟, 'd'=天)
    freq = args.freq

    # =========================================================================
    # Step 4: 根据任务类型，使用不同的初始化方式
    # =========================================================================
    
    # -------------------------------------------------------------------------
    # 任务类型 1: 异常检测 (Anomaly Detection)
    # -------------------------------------------------------------------------
    if args.task_name == 'anomaly_detection':
        drop_last = False
        
        # 创建异常检测数据集
        # 异常检测不需要 pred_len，只需要滑动窗口 (win_size)
        data_set = Data(
            args = args,
            root_path=args.root_path,      # 数据根目录
            win_size=args.seq_len,          # 滑动窗口大小 = 序列长度
            flag=flag,                      # train/val/test
        )
        print(flag, len(data_set))  # 打印数据集大小
        
        # 创建 DataLoader
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,   # 多进程加载数据
            drop_last=drop_last)
        
        return data_set, data_loader
    
    # -------------------------------------------------------------------------
    # 任务类型 2: 时序分类 (Classification)
    # -------------------------------------------------------------------------
    elif args.task_name == 'classification':
        drop_last = False
        
        # 创建分类数据集
        data_set = Data(
            args = args,
            root_path=args.root_path,
            flag=flag,
        )

        # 分类任务使用自定义的 collate_fn
        # collate_fn 用于处理变长序列，将它们填充/截断到统一长度
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)  # 统一序列长度
        )
        return data_set, data_loader
    
    # -------------------------------------------------------------------------
    # 任务类型 3: 时序预测（长期/短期预测、插值等）
    # -------------------------------------------------------------------------
    # 这是最常用的任务类型，包括 long_term_forecast, short_term_forecast, imputation 等
    else:
        # M4 数据集特殊处理：不丢弃最后一个 batch
        if args.data == 'm4':
            drop_last = False
        
        # 创建预测数据集
        # 这是最复杂的初始化，需要指定多个参数
        data_set = Data(
            args = args,
            root_path=args.root_path,       # 数据根目录，如 './dataset/'
            data_path=args.data_path,       # 数据文件名，如 'ETTh1.csv'
            flag=flag,                      # train/val/test
            size=[args.seq_len,             # [输入长度, 标签长度, 预测长度]
                  args.label_len,           # 例如: [96, 48, 96] 
                  args.pred_len],           # 表示用96步预测未来96步
            features=args.features,         # 特征类型: 'M'多变量, 'S'单变量, 'MS'多变量预测单变量
            target=args.target,             # 目标变量列名，如 'OT'
            timeenc=timeenc,                # 时间编码方式
            freq=freq,                      # 时间频率
            seasonal_patterns=args.seasonal_patterns  # 季节性模式（用于M4数据集）
        )
        print(flag, len(data_set))  # 打印数据集大小，如 "train 8521"
        
        # 创建 DataLoader
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,          # 每批样本数
            shuffle=shuffle_flag,           # 是否打乱
            num_workers=args.num_workers,   # 数据加载的子进程数
            drop_last=drop_last)            # 是否丢弃不完整的最后一批
        
        return data_set, data_loader


# =============================================================================
#                              补充说明
# =============================================================================
#
# 【数据流向图】
# 
#     CSV 文件 (ETTh1.csv)
#           ↓
#     Dataset 类 (Dataset_ETT_hour)
#           ↓ __getitem__(index)
#     返回单个样本: (seq_x, seq_y, seq_x_mark, seq_y_mark)
#           ↓
#     DataLoader 批量组装
#           ↓
#     训练循环中迭代: for batch in data_loader
#
#
# 【返回的数据格式】(以预测任务为例)
#
#     batch_x:      [batch_size, seq_len, n_features]      输入序列
#     batch_y:      [batch_size, label_len+pred_len, n_features]  输出序列
#     batch_x_mark: [batch_size, seq_len, time_dim]        输入时间特征
#     batch_y_mark: [batch_size, label_len+pred_len, time_dim]   输出时间特征
#
#
# 【如何添加自己的数据集】
#
#     1. 在 data_loader.py 中创建新的 Dataset 类
#     2. 在本文件的 data_dict 中注册: 'my_data': Dataset_MyData
#     3. 运行时使用: --data my_data --data_path my_data.csv
#
# =============================================================================
