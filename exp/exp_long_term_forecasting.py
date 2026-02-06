from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')

# ============================================================================
# 长期时间序列预测实验类
# 该模块实现了时间序列长期预测的完整实验流程，包括模型训练、验证、测试
# 支持多种模型结构、多GPU并行计算、自动混合精度等高级特性
# ============================================================================

class Exp_Long_Term_Forecast(Exp_Basic):
    """长期预测实验类，继承自基础实验类 Exp_Basic"""
    
    def __init__(self, args):
        """初始化长期预测实验类
        
        Args:
            args: 配置参数对象，包含模型名称、设备类型、学习率等信息
        """
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        """构建预测模型
        
        Returns:
            model: 构建好的深度学习模型，支持单GPU和多GPU并行计算
        """
        # 根据配置从模型字典中加载指定的模型，并转换为float32精度
        model = self.model_dict[self.args.model](self.args).float()

        # 如果开启多GPU模式，使用 DataParallel 进行分布式计算
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        """获取指定类型的数据集和数据加载器
        
        Args:
            flag: 数据类型标记，可选值为 'train'（训练）、'val'（验证）或 'test'（测试）
            
        Returns:
            data_set: 数据集对象，包含数据和归一化/反归一化信息
            data_loader: PyTorch 数据加载器，用于批量加载数据
        """
        # 调用数据工厂函数生成相应的数据集和加载器
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """选择模型优化器
        
        Returns:
            model_optim: Adam 优化器实例
        """
        # 使用 Adam 优化器进行模型参数优化，初始学习率从配置中读取
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """选择损失函数
        
        Returns:
            criterion: 均方误差损失函数（MSE Loss）
        """
        # 使用平均平方误差作为损失函数，对回归任务优化
        criterion = nn.MSELoss()
        return criterion
 

    def vali(self, vali_data, vali_loader, criterion):
        """验证函数：计算验证集上的损失
        
        Args:
            vali_data: 验证数据集对象
            vali_loader: 验证数据加载器
            criterion: 损失函数
            
        Returns:
            total_loss: 验证集上的平均损失值
        """
        total_loss = []
        # 将模型设为评估模式（禁用 dropout, batchnorm 等训练特定操作）
        self.model.eval()
        # 禁用梯度计算以节省显存和加速验证过程
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                # 将输入数据转移到指定设备（GPU/CPU）
                batch_x = batch_x.float().to(self.device)  # 编码器输入序列
                batch_y = batch_y.float()  # 目标输出序列

                batch_x_mark = batch_x_mark.float().to(self.device)  # 编码器时间标记信息
                batch_y_mark = batch_y_mark.float().to(self.device)  # 解码器时间标记信息

                # 构造解码器输入：在预测长度位置用零填充，然后拼接前 label_len 个真实值
                # 这是 Seq2Seq 模型的标准做法
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # 执行编码器-解码器前向传播
                # 如果启用自动混合精度（AMP），使用 FP16 加速计算
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # 根据特征类型选择维度：
                # 'MS' (多元预测单变量目标) 时取最后一列，其他情况取所有列
                f_dim = -1 if self.args.features == 'MS' else 0
                # 提取最后 pred_len 个预测时间步
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # 分离张量，停止梯度追踪
                pred = outputs.detach()
                true = batch_y.detach()

                # 计算 MSE 损失
                loss = criterion(pred, true)

                total_loss.append(loss.item())
        
        # 计算平均损失
        total_loss = np.average(total_loss)
        # 恢复模型为训练模式
        self.model.train()
        return total_loss

    def train(self, setting):
        """完整的模型训练函数
        
        Args:
            setting: 实验配置字符串，用于保存检查点和日志
            
        Returns:
            model: 训练好的模型
        """
        # 加载训练、验证、测试数据
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # 创建检查点保存目录
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # 记录训练开始时间
        time_now = time.time()

        # 获取每个 epoch 的训练步数
        train_steps = len(train_loader)
        # 初始化早停机制：当验证集性能不再改善时停止训练
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 选择优化器和损失函数
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 如果使用自动混合精度，初始化梯度缩放器
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # 按 epoch 循环进行训练
        for epoch in range(self.args.train_epochs):
            iter_count = 0  # 迭代计数器
            train_loss = []  # 训练损失列表

            # 设置模型为训练模式
            self.model.train()
            epoch_time = time.time()  # 记录 epoch 开始时间
            
            # 遍历训练数据加载器中的每个批次
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                # 清空梯度缓存
                model_optim.zero_grad()
                # 将数据转移到计算设备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 构造解码器输入
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 前向传播：执行编码器-解码器预测
                if self.args.use_amp:
                    # 自动混合精度模式：部分计算使用 FP16
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        # 选择预测维度
                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                        # 计算损失
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    # 标准 FP32 计算模式
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    # 选择预测维度
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    # 计算损失
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # 每 100 个迭代步输出一次训练进度
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    # 计算训练速度和剩余时间
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # 反向传播和参数更新
                if self.args.use_amp:
                    # AMP 模式：使用梯度缩放器进行反向传播
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    # 标准模式：直接计算梯度和更新参数
                    loss.backward()
                    model_optim.step()

            # 输出该 epoch 的训练耗时
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            # 计算平均训练损失
            train_loss = np.average(train_loss)
            # 在验证集和测试集上评估模型
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            # 输出三个数据集上的性能指标
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            
            # 早停检查：如果验证集性能有改善，保存模型
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 根据 epoch 调整学习率（通常采用递减策略）
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        # 加载最佳性能的模型（由早停机制保存）
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        """模型测试函数：在测试集上评估模型，计算各项评价指标
        
        Args:
            setting: 实验配置字符串
            test: 是否加载已保存的模型（1表示加载，0表示使用当前模型）
        """
        # 加载测试数据
        test_data, test_loader = self._get_data(flag='test')
        # 如果需要，加载训练好的模型检查点
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []  # 存储所有预测值
        trues = []  # 存储所有真实值
        # 创建可视化结果保存目录
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 设置模型为评估模式
        self.model.eval()
        # 禁用梯度计算
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                # 将数据转移到计算设备
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 构造解码器输入
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # 前向传播
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # 提取预测维度
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]  # 提取最后 pred_len 个预测
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)  # 提取最后 pred_len 个真实值
                
                # 转换为 numpy 格式
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                # 如果数据已归一化，需要反归一化到原始值域
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    # 处理预测和真实值维度不匹配的情况（多元预测单变量目标）
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    # 执行反归一化
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                # 选择预测维度
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                # 累积预测和真实值
                preds.append(pred)
                trues.append(true)
                
                # 每 20 个批次保存一张预测对比图
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    # 拼接输入和预测/真实值进行可视化
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    # 保存预测对比图
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        # 合并所有批次的预测和真实值
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        # 重新整形数据
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # 创建结果保存目录
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # 可选：计算 DTW（动态时间规整）距离，用于衡量时间序列相似性
        if self.args.use_dtw:
            dtw_list = []
            # 定义距离函数：曼哈顿距离
            manhattan_distance = lambda x, y: np.abs(x - y)
            # 对每条时间序列计算 DTW 距离
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)  # 预测序列
                y = trues[i].reshape(-1, 1)  # 真实序列
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()  # 计算平均 DTW 距离
        else:
            dtw = 'Not calculated'

        # 计算回归评价指标：MAE、MSE、RMSE、MAPE、MSPE
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        
        # 将评价指标追加到结果文件
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        # 保存评价指标、预测值和真实值为 numpy 文件
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
