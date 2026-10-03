import torch
from Dynamics import AcceleratedMovementModel

class AdaptiveExtendedKalmanFilter:
    def __init__(self, initial_state, initial_covariance,
                 process_noise, measurement_noise, dt, device = 'cuda'):
        """
        initial_state: [B,state_dim] or [state_dim]
        initial_covariance: [B,state_dim,state_dim] or [state_dim,state_dim]
        process_noise: [B,state_dim,state_dim] or [state_dim,state_dim]
        measurement_noise: [B,obs_dim,obs_dim] or [obs_dim,obs_dim]
        """
        if device is None:
            device = torch.device('cpu')
        self.device = device

        if isinstance(dt, torch.Tensor):
            dt_t = dt.to(device).item() if dt.numel() == 1 else float(dt)
        else:
            dt_t = float(dt)
        
        # 获取AEKF模型参数
        Dynamics_model = AcceleratedMovementModel(dt_t, device=self.device)
        self.state_dim, self.obs_dim, self.acc_dim, self.F, _, self.B_mat = Dynamics_model.get_dynamics()

        # 状态
        self.state = self._to_batch(initial_state, (self.state_dim,), device)                            # [B,state_dim]
        self.covariance = self._to_batch(initial_covariance, (self.state_dim, self.state_dim), device)   # [B,state_dim,state_dim]
        self.covariance_pre = self._to_batch(initial_covariance, (self.state_dim, self.state_dim), device)   # [B,state_dim,state_dim]
        self.Q = self._to_batch(process_noise, (self.state_dim, self.state_dim), device)                 # [B,state_dim,state_dim]
        self.R = self._to_batch(measurement_noise, (self.obs_dim, self.obs_dim), device)                 # [B,obs_dim,obs_dim]

        B = self.state.shape[0]
        self.innovation = torch.zeros(B, self.obs_dim, device=device)           # [B,obs_dim]
        self.innovation_diff = torch.zeros(B, self.obs_dim, device=device)      # [B,obs_dim]
        self.predict_error = torch.zeros(B, self.state_dim, device=device)      # [B,state_dim]
        self.H = torch.zeros(B, self.obs_dim, self.state_dim, dtype=torch.float32, device=device)  # [B,obs_dim,state_dim]

        # 预分配预测状态
        self.state_pre = self.state.clone()

    @staticmethod
    def _to_batch(x, single_shape, device):
        """
        把 [*], [B,*] 或 numpy 转成 [B,*] 的 torch.Tensor
        """
        if isinstance(x, torch.Tensor):
            t = x.to(device)
        else:
            t = torch.as_tensor(x, dtype=torch.float32, device=device)

        if t.dim() == 1:          # [n]
            assert t.shape[0] == single_shape[0]
            t = t.unsqueeze(0)    # [1,n]
        elif t.dim() == 2 and t.shape == single_shape:
            t = t.unsqueeze(0)    # [1,n,n]

        return t  # [B,*] or already [B,*]

    def predict(self, u, adaptive_Q):
        """
        预测步
        dt: 标量 float 或标量 tensor
        u: [B,3] 或 None
        adaptive_Q: [B,state_dim,state_dim] 或 [state_dim,state_dim] 或 None
        """
        B = self.state.shape[0]
        device = self.device

        # 预测状态
        if u is None:
            self.state_pre = (self.state @ self.F.T)                              # [B,state_dim]
        else:
            u = torch.as_tensor(u, dtype=torch.float32, device=device)            # [B,acc_dim]
            self.state_pre = (self.state @ self.F.T) + (u @ self.B_mat.T)         # [B,state_dim]

        # 自适应 Q
        if adaptive_Q is not None:
            self.Q = self._to_batch(adaptive_Q, single_shape=(self.state_dim, self.state_dim), device=self.device)

        # 预测协方差：F @ P @ F^T + Q，做 batched 乘法
        F_batch = self.F.unsqueeze(0).expand(B, self.state_dim, self.state_dim)   # [B,state_dim,state_dim]
        P_pred = F_batch @ self.covariance @ F_batch.transpose(1, 2)              # [B,state_dim,state_dim]
        self.covariance_pre = P_pred + self.Q                                         # [B,state_dim,state_dim]

    def update(self, z, adaptive_R):
        """
        更新步
        z: [B,4]
        adaptive_R: [B,4,4] 或 [4,4] 或 None
        """
        device = self.device
        B = self.state.shape[0]
    
        z = torch.as_tensor(z, dtype=torch.float32, device=device)  # [B,obs_dim]
    
        # 提取位置分量
        p_x = self.state[:, 0]  # [B]
        p_y = self.state[:, 1]
        p_z = self.state[:, 2]
        eps = 1e-8
        p = torch.sqrt(p_x**2 + p_y**2 + p_z**2 + eps)  # [B]

        # 构造 H，每个 batch 一个 4x6 矩阵
        # 第一行 = [p_x/p , p_y/p , p_z/p , 0 , 0 , 0]
        e1 = torch.stack([p_x/p, p_y/p, p_z/p], dim=1)  # [B,3]
        row0 = torch.cat([e1, torch.zeros(B,3, device=device)], dim=1)  # [B,6]
        # 第二行 = [0 0 0 1 0 0]
        row1 = torch.tensor([0,0,0,1,0,0], device=device).expand(B,6)
        # 第三行 = [0 0 0 0 1 0]
        row2 = torch.tensor([0,0,0,0,1,0], device=device).expand(B,6)
        # 第四行 = [0 0 0 0 0 1]
        row3 = torch.tensor([0,0,0,0,0,1], device=device).expand(B,6)
        H = torch.stack([row0, row1, row2, row3], dim=1)  # [B,4,6]

        # 自适应 R
        if adaptive_R is not None:
            self.R = self._to_batch(adaptive_R, single_shape=(self.obs_dim, self.obs_dim), device=self.device)  # [B,obs_dim,obs_dim]

        # S = H P H^T + R
        H_t = H.transpose(1, 2)                                   # [B,obs_dim,state_dim]
        S = H @ self.covariance_pre @ H_t + self.R                    # [B,obs_dim,obs_dim]
        
        # K = P H^T S^{-1}  pinv 提高数值稳定性
        S_inv = torch.linalg.pinv(S)                              # [B,obs_dim,obs_dim]
        K = self.covariance_pre @ H_t @ S_inv                         # [B,state_dim,obs_dim]

        # 计算预测量测 y_pre
        p_x_pre = self.state_pre[:, 0]
        p_y_pre = self.state_pre[:, 1]
        p_z_pre = self.state_pre[:, 2]
        p_pre = torch.sqrt(p_x_pre**2 + p_y_pre**2 + p_z_pre**2 + eps)  # [B]

        y_pre = torch.stack([
            p_pre,
            self.state_pre[:, 3],
            self.state_pre[:, 4],
            self.state_pre[:, 5]
        ], dim=-1)  # [B,obs_dim]

        # 新息
        y = z - y_pre  # [B,obs_dim]

        I_state_dim = torch.eye(self.state_dim, dtype=torch.float32, device=device).unsqueeze(0).expand(B, self.state_dim, self.state_dim)
        I_obs_dim = torch.eye(self.obs_dim, dtype=torch.float32, device=device).unsqueeze(0).expand(B, self.obs_dim, self.obs_dim)

        # 更新状态
        self.state = self.state_pre + (K @ y.unsqueeze(-1)).squeeze(-1)  # [B,state_dim]
        x_res = self.state - self.state_pre
        x_res_cov = K @ S @ K.transpose(1,2)
        x_res_cov = 0.5 * (x_res_cov + x_res_cov.transpose(1, 2))   # 对称化
        x_res_cov = x_res_cov + 1e-6 * torch.eye(x_res_cov.size(-1), device=x_res_cov.device).unsqueeze(0)

        # 更新协方差： (I - K H) P
        KH = K @ H                                                       # [B,self.state_dim,self.state_dim]
        HK = H @ K
        
        S_post = (I_obs_dim - HK) @ S @ (I_obs_dim - HK).transpose(1,2)
        self.covariance = (I_state_dim - KH) @ self.covariance_pre                     # [B,self.state_dim,self.state_dim]
        #self.covariance = 0.5 * (self.covariance + self.covariance.transpose(1, 2))
        
        #print(self.covariance)

        # 预测误差
        self.predict_error = self.state - self.state_pre                 # [B,self.state_dim]

        # 更新后的量测 y_post 用新的 state
        p_x_post = self.state[:, 0]
        p_y_post = self.state[:, 1]
        p_z_post = self.state[:, 2]
        p_post = torch.sqrt(p_x_post**2 + p_y_post**2 + p_z_post**2 + eps)

        y_post = torch.stack([
            p_post,
            self.state[:, 3],
            self.state[:, 4],
            self.state[:, 5]
        ], dim=-1)  # [B,obs_dim]

        # 残差
        residual = z - y_post  # [B,obs_dim]  

        # 新息
        innov_pre = self.innovation
        self.innovation = z - y_pre  # [B,obs_dim]

        # 新息差分 
        self.innovation_diff = self.innovation - innov_pre  # [B,obs_dim]

        self.H = H

        return S, residual, S_post, z - y_pre, x_res, x_res_cov

    def get_state(self):
        return self.state                   # [B,state_dim]

    def get_covariance(self):
        return self.covariance              # [B,state_dim,state_dim]

    def get_QandR(self):
        return self.Q, self.R               # [B,state_dim,state_dim], [B,obs_dim,obs_dim]

    def get_innovation(self):
        return self.innovation              # [B,obs_dim]
    
    def get_innovation_diff(self):
        return self.innovation_diff         # [B,obs_dim]

    def get_predict_error(self):
        return self.predict_error           # [B,state_dim]
    
    def get_H(self):
        return self.H                       

        return S, residual, S_post, z - y_pre, x_res, x_res_cov

    def get_state(self):
        return self.state                   # [B,state_dim]

    def get_covariance(self):
        return self.covariance              # [B,state_dim,state_dim]

    def get_QandR(self):
        return self.Q, self.R               # [B,state_dim,state_dim], [B,obs_dim,obs_dim]

    def get_innovation(self):
        return self.innovation              # [B,obs_dim]
    
    def get_innovation_diff(self):
        return self.innovation_diff         # [B,obs_dim]

    def get_predict_error(self):
        return self.predict_error           # [B,state_dim]
    
    def get_H(self):
        return self.H                       
