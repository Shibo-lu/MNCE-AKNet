import torch

# AEKF Dynamics 1: 加速运动目标
class AcceleratedMovementModel:
    def __init__(self, dt, device = 'cuda'):

        self.state_dim = 6
        self.obs_dim = 4
        self.acc_dim = 3

        # 状态转移矩阵 F 和控制矩阵 B_mat
        self.F = torch.tensor([
            [1, 0, 0, dt,   0,    0],
            [0, 1, 0, 0,    dt,   0],
            [0, 0, 1, 0,    0,   dt],
            [0, 0, 0, 1,    0,    0],
            [0, 0, 0, 0,    1,    0],
            [0, 0, 0, 0,    0,    1]
        ], dtype=torch.float32, device=device)  # [6,6]

        # 定义一个失配矩阵 
        theta = torch.tensor(10.0) * torch.pi / 180  # 弧度制
        self.F_mismatch = torch.tensor([
            [1, 0, 0, dt * torch.cos(theta), - dt * torch.sin(theta), 0],
            [0, 1, 0, dt * torch.sin(theta), dt * torch.cos(theta), 0],
            [0, 0, 1, 0,    0,   dt],
            [0, 0, 0, 1,    0,    0],
            [0, 0, 0, 0,    1,    0],
            [0, 0, 0, 0,    0,    1]
        ], dtype=torch.float32, device=device)  # [6,6]

        self.B_mat = torch.tensor([
            [0.5 * dt**2, 0,                0               ],
            [0,             0.5 * dt**2,    0               ],
            [0,             0,                0.5 * dt**2   ],
            [dt,          0,                0               ],
            [0,             dt,             0               ],
            [0,             0,                dt            ]
        ], dtype=torch.float32, device=device)  # [6,3]
    
    def get_dynamics(self):
        return self.state_dim, self.obs_dim, self.acc_dim, self.F, self.F_mismatch, self.B_mat