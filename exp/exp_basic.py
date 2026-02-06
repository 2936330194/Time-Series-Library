import os
import torch
import importlib
import pkgutil  

# 只需将模型文件放在 models/ 文件夹下
# 例如：models/Transformer.py, models/LSTM.py 等
# 所有模型将被自动检测，可以通过指定其名称使用

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        
        # -------------------------------------------------------
        #  自动生成模型映射
        # -------------------------------------------------------
        model_map = self._scan_models_directory()

        # 使用智能字典
        self.model_dict = LazyModelDict(model_map)

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _scan_models_directory(self):
        """
        自动扫描 models 文件夹中的所有 .py 文件
        """
        model_map = {}
        models_dir = 'models'

        # 遍历 'models' 目录中的所有文件
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                # 忽略 __init__.py 和非 .py 文件
                if filename.endswith('.py') and filename != '__init__.py':
                    # 移除 .py 扩展名以获取模块名
                    module_name = filename[:-3]
                    
                    # 构建完整的导入路径
                    full_path = f"{models_dir}.{module_name}"
                    
                    # 加载字典：{'Transformer': 'models.Transformer'}
                    model_map[module_name] = full_path
        
        return model_map

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu and self.args.gpu_type == 'cuda':
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        elif self.args.use_gpu and self.args.gpu_type == 'mps':
            device = torch.device('mps')
            print('Use GPU: mps')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


class LazyModelDict(dict):
    """
    智能懒加载字典
    """
    def __init__(self, model_map):
        self.model_map = model_map
        super().__init__()

    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        
        if key not in self.model_map:
            raise NotImplementedError(f"Model [{key}] not found in 'models' directory.")
            
        module_path = self.model_map[key]
        try:
            print(f"🚀 Lazy Loading: {key} ...") 
            module = importlib.import_module(module_path)
        except ImportError as e:
            print(f"❌ Error: Failed to import model [{key}]. Dependencies missing?")
            raise e

        # 尝试查找模型类
        if hasattr(module, 'Model'):
            model_class = module.Model
        elif hasattr(module, key):
            model_class = getattr(module, key)
        else:
            raise AttributeError(f"Module {module_path} has no class 'Model' or '{key}'")

        self[key] = model_class
        return model_class

