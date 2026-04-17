import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowC3D2(nn.Module):
    """
    A shallow 3D CNN model, similar to C3D, for video-like (3D image) data.
    """
    def __init__(self, num_classes: int = 1):  # Binary classification output (1 logit for BCEWithLogitsLoss)
        super(ShallowC3D2, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv1l = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=1)
        self.pool1l = nn.MaxPool3d(kernel_size=(6, 2, 2), stride=(6, 2, 2))

        self.conv2l = nn.Conv3d(64, 32, kernel_size=(3, 3, 3), padding=1)
        self.pool2l = nn.MaxPool3d(kernel_size=(6, 2, 2), stride=(6, 2, 2))

        self.ax_stem1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.ax_stem_bn1 = nn.BatchNorm2d(16)
        self.dropout2d1 = nn.Dropout2d(0.5)

        self.ax_stem2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.ax_stem_bn2 = nn.BatchNorm2d(16)
        self.dropout2d2 = nn.Dropout2d(0.2)

        self.ax_dw = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1, groups=16)
        self.ax_pw = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

        self.fc1 = nn.Linear(1280, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x_input: torch.Tensor, va_logmar_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ShallowC3D2 model.

        Args:
            x_input (torch.Tensor): Input image tensor.
            va_logmar_input (torch.Tensor): VAlogmar input tensor.

        Returns:
            torch.Tensor: Output logits.
        """
        x = x_input.unsqueeze(1)
        x = F.relu(self.conv1l(x))
        x = self.pool1l(x)

        x = F.relu(self.conv2l(x))
        x = self.pool2l(x)

        x = torch.squeeze(x, dim=2)

        x = F.relu(self.ax_stem_bn1(self.dropout2d1(self.ax_stem1(x))))
        x = self.max_pool(x)

        x = self.ax_dw(x)
        x = self.max_pool(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x