import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np

# 从新开始训练resnet参数
class ImageCritic(nn.Module):
    def __init__(self, camera_names, state_dim=20, hidden_dim=256):
        super().__init__()
        self.camera_names = camera_names
        
        # 🌟 优化 1: 强烈建议使用预训练权重！
        # 在 RL 中从头训 ResNet 极难收敛。使用 ImageNet 预训练特征能将训练速度提升数倍。
        self.visual_encoders = nn.ModuleDict()
        for cam in camera_names:
            # 引入 ImageNet 默认权重 不使用优化权重则设置weights=none
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) 
            self.visual_encoders[cam] = nn.Sequential(*list(resnet.children())[:-1])
            
        # 🌟 优化 2: 匹配预训练权重的标准归一化
        # 因为前置代码你只除以了 255.0，这里补上 ImageNet 期望的 Mean 和 Std
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        # 🌟 优化 3: 为状态特征添加 LayerNorm
        # 图像特征是 512*相机数 维，状态特征只有 hidden_dim 维。
        # 加上 LayerNorm 防止数值较小的 State 特征被庞大的视觉特征“淹没”
        self.state_dim = state_dim
        if self.state_dim is not None: # 如果状态特征不为空
            self.state_encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            )
            total_feature_dim = len(camera_names) * 512 + hidden_dim
        else:
            self.state_encoder = None
            total_feature_dim = len(camera_names) * 512

        self.mlp = nn.Sequential(
            nn.Linear(total_feature_dim, 512),
            nn.LayerNorm(512), # 加入 LayerNorm 稳定 PPO 训练的高级技巧
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # 🌟 优化 4: PPO 祖传秘方 —— 正交初始化
        self._apply_orthogonal_init()

    def _apply_orthogonal_init(self):
        """对 MLP 层应用正交初始化，并在最后一层将权重极度缩小"""
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        # 🌟 极其关键：最后一层 Gain 设为 0.01，确保初始 V(s) 接近 0。
        # 这样第一轮更新时的 Advantage 才不会因为 Critic 的瞎猜而爆炸。
        nn.init.orthogonal_(self.mlp[-1].weight, gain=0.01)

    def forward(self, batch):
        features = []
        
        # 提取图像特征
        for cam in self.camera_names:
            img_tensor = batch[f'observation.images.{cam}'] # 期望输入: [BS, C, H, W], 数值域 0.0~1.0
            
            # 应用标准化
            img_tensor = self.normalize(img_tensor)
            
            # 🌟 优化 5: 用 flatten 替代连续的 squeeze
            # ResNet 输出 [BS, 512, 1, 1]。flatten(start_dim=1) 更安全、鲁棒。
            feat = self.visual_encoders[cam](img_tensor).flatten(start_dim=1)
            features.append(feat)
            
        # 提取状态特征
        if self.state_dim is not None:
            state_tensor = batch['observation.state'] 
            state_feat = self.state_encoder(state_tensor)
            features.append(state_feat)
        
        # 拼接所有特征
        concat_features = torch.cat(features, dim=-1)
        
        # 计算价值 V(s)
        value = self.mlp(concat_features) # [BS, 1]
        
        return value.squeeze(-1) # 返回 [BS]
    
# 和actor共享视觉底座
class SharedFeatureCritic(nn.Module):
    def __init__(self, global_cond_dim):
        """
        纯净版共享 Critic：没有卷积，没有状态编码器。
        参数 global_cond_dim: Actor 吐出的全局特征向量维度
        """
        super().__init__()
        
        # 核心 MLP 评价网络 (由于输入已经是高维融合特征，直接上 512)
        self.mlp = nn.Sequential(
            nn.Linear(global_cond_dim, 512),
            nn.LayerNorm(512), 
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self._apply_orthogonal_init()

    def _apply_orthogonal_init(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.mlp[-1].weight, gain=0.01)

    def forward(self, global_cond):
        """期望输入: Actor 提取好的带有梯度树的特征向量 [BS, global_cond_dim]"""
        value = self.mlp(global_cond) # [BS, 1]
        return value.squeeze(-1)      # [BS]
