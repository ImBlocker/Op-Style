import torch
import torch.nn as nn
#
# class ActionNetworkBase(nn.Module):
#     def __init__(self, n_channels: int, in_dim: int, intermediate_size: int = 256, action_num = 4) -> None:
#         super().__init__()
#         self.n_channels = n_channels
#         self.in_dim = in_dim
#         self.intermediate_size = intermediate_size
#         self.action_num = action_num
#
#         self.fc = nn.Sequential(
#             nn.Conv2d(self.n_channels, 64, kernel_size=3, stride=2, padding=1), # 8
#             nn.LeakyReLU(inplace=True),
#             nn.Dropout2d(p=0.1),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# 4
#             nn.LeakyReLU(inplace=True),
#             nn.Dropout2d(p=0.1),
#             nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),# 2
#             nn.LeakyReLU(inplace=True),
#             nn.Dropout2d(p=0.1),
#             nn.Flatten(), # 128, 2, 2
#             nn.Linear(512, self.intermediate_size),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.intermediate_size, self.action_num)
#         )
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         action = self.fc(x.to(torch.float))
#         return action



class HighActionNetwork(nn.Module):
    def __init__(self, n_channels: int, in_dim: int, intermediate_size: int = 256, action_num = 4) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.in_dim = in_dim
        self.intermediate_size = intermediate_size
        self.action_num = action_num

        self.fc = nn.Sequential(
            nn.Conv2d(self.n_channels, 64, kernel_size=3, stride=1, padding=1), # 16
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# 8
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),# 8
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 4
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # 4
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 2
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Flatten(), # 512, 2, 2
            nn.Linear(512*4, self.intermediate_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.intermediate_size, self.action_num)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        action = self.fc(x.to(torch.float))
        return action


class MiddleActionNetwork(nn.Module):
    def __init__(self, n_channels: int, in_dim: int, intermediate_size: int = 256, action_num = 4) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.in_dim = in_dim
        self.intermediate_size = intermediate_size
        self.action_num = action_num

        self.fc = nn.Sequential(
            nn.Conv2d(self.n_channels, 64, kernel_size=3, stride=1, padding=1),  # 16
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 8
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),# 4
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),# 2
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Flatten(), # 128, 2, 2
            nn.Linear(512, self.intermediate_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.intermediate_size, self.action_num)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        action = self.fc(x.to(torch.float))
        return action

class LowActionNetwork_simplest(nn.Module):
    def __init__(self, n_channels: int, in_dim: int, intermediate_size: int = 256, action_num=4) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.in_dim = in_dim
        self.intermediate_size = intermediate_size
        self.action_num = action_num

        self.fc = nn.Sequential(
            nn.Conv2d(self.n_channels, 14, kernel_size=3, stride=2, padding=1),  # 8
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Conv2d(14, 14, kernel_size=3, stride=2, padding=1),  # 4
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.5),
            nn.Flatten(),
            nn.Linear(14 * 16, self.intermediate_size),
            nn.ReLU(inplace=True))


        self.fc1 = nn.Linear(7, self.intermediate_size)

        self.fc2 = nn.Linear(2 * self.intermediate_size, self.action_num)

        self.fc3 = nn.Linear(self.intermediate_size, self.action_num)


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        action1 = self.fc(x.to(torch.float))
        action2 = self.fc1(y.to(torch.float))
        action = self.fc2(torch.cat([action1, action2], dim=1))

        return action

    def forward_without_y(self, x: torch.Tensor):
        action = self.fc3(self.fc(x.to(torch.float)))
        return action


class LowActionNetwork_simple(nn.Module):
    def __init__(self, n_channels: int, in_dim: int, intermediate_size: int = 256, action_num=4) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.in_dim = in_dim
        self.intermediate_size = intermediate_size
        self.action_num = action_num

        self.fc = nn.Sequential(
            nn.Conv2d(self.n_channels, 28, kernel_size=3, stride=2, padding=1),  # 8
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(28, 64, kernel_size=3, stride=2, padding=1),  # 4
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.Flatten(),  # 128, 4, 4
            nn.Linear(64 * 16, self.intermediate_size),
            nn.ReLU(inplace=True))


        self.fc1 = nn.Linear(7, self.intermediate_size)

        self.fc2 = nn.Linear(2 * self.intermediate_size, self.action_num)

        self.fc3 = nn.Linear(self.intermediate_size, self.action_num)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        action1 = self.fc(x.to(torch.float))
        action2 = self.fc1(y.to(torch.float))
        action = self.fc2(torch.cat([action1, action2], dim=1))

        return action

    def forward_without_y(self, x: torch.Tensor):
        action = self.fc3(self.fc(x.to(torch.float)))
        return action

class LowActionNetwork_complex(nn.Module):
    def __init__(self, n_channels: int, in_dim: int, intermediate_size: int = 256, action_num=4) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.in_dim = in_dim
        self.intermediate_size = intermediate_size
        self.action_num = action_num

        self.fc = nn.Sequential(
            nn.Conv2d(self.n_channels, 28, kernel_size=3, stride=2, padding=1),  # 8
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(28, 64, kernel_size=3, stride=2, padding=1),  # 4
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Flatten(),  # 128, 4, 4
            nn.Linear(64 * 16, self.intermediate_size),
            nn.ReLU(inplace=True))


        self.fc1 = nn.Linear(7, self.intermediate_size)

        self.fc2 = nn.Linear(2 * self.intermediate_size, self.action_num)

        self.fc3 = nn.Linear(self.intermediate_size, self.action_num)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        action1 = self.fc(x.to(torch.float))
        action2 = self.fc1(y.to(torch.float))
        action = self.fc2(torch.cat([action1, action2], dim=1))

        return action

    def forward_without_y(self, x: torch.Tensor):
        action = self.fc3(self.fc(x.to(torch.float)))
        return action