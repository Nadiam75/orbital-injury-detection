import torch
import torch.nn as nn
import torch.nn.functional as F

class AxCorCNN_Model(nn.Module):
    """
    A CNN model with separate axial and coronal processing branches,
    concatenated features, and a final classification head.

    Args:
        mode (str): Mode for the model (e.g., 'ax').
        multiclass (bool): If True, output is for multi-class classification. Defaults to False.
    """
    def __init__(self, mode: str, multiclass: bool = False):
        super(AxCorCNN_Model, self).__init__()
        self.mode = mode
        self.multiclass = multiclass
        self.filters = 16

        # Common layers
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

        # Build branches
        self._build_coronal_branch()
        self._build_axial_branch()
        self._build_final_head()

    def _build_coronal_branch(self):
        """
        Defines the layers for the coronal image processing branch.
        Input channels for cor_stem1 should be 32 based on the input data shape.
        """
        stage1 = 64
        stage2 = 32
        stage3 = 16 # Renamed from original's stage3 (16) to avoid confusion.

        self.cor_stem1 = nn.Conv2d(32, stage1, kernel_size=3, stride=1, padding=1) # Input channels 32
        self.cor_stem_bn1 = nn.BatchNorm2d(stage1)
        self.cor_dropout2d1 = nn.Dropout2d(0.5)

        self.cor_stem2 = nn.Conv2d(stage1, stage2, kernel_size=3, stride=1, padding=1)
        self.cor_stem_bn2 = nn.BatchNorm2d(stage2)
        self.cor_dropout2d2 = nn.Dropout2d(0.5)

        self.cor_stem3 = nn.Conv2d(stage2, stage3, kernel_size=3, stride=1, padding=1)
        self.cor_stem_bn3 = nn.BatchNorm2d(stage3)
        self.cor_dropout2d3 = nn.Dropout2d(0.5)

        self.cor_stem4 = nn.Conv2d(stage3, 64, kernel_size=3, stride=1, padding=1)
        self.cor_stem_bn4 = nn.BatchNorm2d(64)
        self.cor_dropout2d4 = nn.Dropout2d(0.2)

        self.cor_conv = nn.Conv2d(self.filters, self.filters, kernel_size=3, stride=1, padding=1, groups=self.filters)
        self.cor_conv_bn = nn.BatchNorm2d(self.filters)

        self.cor_dw = nn.Conv2d(in_channels=self.filters, out_channels=self.filters,
                                 kernel_size=(32, 32), stride=(32, 32), groups=self.filters, bias=False)

        self.fc_cor = nn.Linear(self.filters, 10) # Output 10 features for concatenation

    def _build_axial_branch(self):
        """
        Defines the layers for the axial image processing branch.
        Input channels for ax_stem1 should be 40 based on input data shape.
        """
        stage1 = 64
        stage2 = 32
        stage3 = 16

        self.ax_stem1 = nn.Conv2d(40, stage1, kernel_size=3, stride=1, padding=1) # Input channels 40
        self.ax_stem_bn1 = nn.BatchNorm2d(stage1)
        self.ax_dropout2d1 = nn.Dropout2d(0.5)

        self.ax_stem2 = nn.Conv2d(stage1, stage2, kernel_size=3, stride=1, padding=1)
        self.ax_stem_bn2 = nn.BatchNorm2d(stage2)
        self.ax_dropout2d2 = nn.Dropout2d(0.5)

        self.ax_stem3 = nn.Conv2d(stage2, stage3, kernel_size=3, stride=1, padding=1)
        self.ax_stem_bn3 = nn.BatchNorm2d(stage3)
        self.ax_dropout2d3 = nn.Dropout2d(0.5)

        self.ax_stem4 = nn.Conv2d(stage3, 64, kernel_size=3, stride=1, padding=1)
        self.ax_stem_bn4 = nn.BatchNorm2d(64)
        self.ax_dropout2d4 = nn.Dropout2d(0.2)

        self.ax_conv = nn.Conv2d(self.filters, self.filters, kernel_size=3, stride=1, padding=1, groups=self.filters)
        self.ax_conv_bn = nn.BatchNorm2d(self.filters)

        self.ax_dw = nn.Conv2d(in_channels=self.filters, out_channels=self.filters,
                                 kernel_size=(19, 15), stride=(19, 15), groups=self.filters, bias=False)
        self.fc_ax = nn.Linear(self.filters, 10) # Output 10 features for concatenation

    def _build_final_head(self):
        """
        Defines the final classification head of the model.
        Input to fc_final is 20 (10 from axial + 10 from coronal).
        """
        self.fc_final = nn.Linear(20, 1)

    def forward(self, ax_input: torch.Tensor, cor_input: torch.Tensor,
                va_logmar_input: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Forward pass of the AxCorCNN_Model.

        Args:
            ax_input (torch.Tensor): Axial image input tensor (B, C_ax, H, W).
            cor_input (torch.Tensor): Coronal image input tensor (B, C_cor, H, W).
            va_logmar_input (torch.Tensor): VAlogmar input tensor (B, 1).
            training (bool): Flag indicating if the model is in training mode.

        Returns:
            torch.Tensor: The output logits from the final classification layer (B, 1).
        """
        # Coronal Branch
        cor_x = F.relu(self.cor_stem_bn1(self.cor_dropout2d1(self.cor_stem1(cor_input))))
        cor_x = self.max_pool(cor_x)
        cor_x = F.relu(self.cor_stem_bn2(self.cor_dropout2d2(self.cor_stem2(cor_x))))
        cor_x = self.max_pool(cor_x)
        cor_x = F.relu(self.cor_stem_bn3(self.cor_stem3(cor_x)))
        cor_x = F.relu(self.cor_conv_bn(self.cor_conv(cor_x)))
        cor_x = self.max_pool(cor_x)
        cor_x = F.relu(self.cor_dw(cor_x))
        cor_x = torch.flatten(cor_x, 1)
        cor_x = F.relu(self.fc_cor(cor_x))

        # Axial Branch
        ax_x = F.relu(self.ax_stem_bn1(self.ax_dropout2d1(self.ax_stem1(ax_input))))
        ax_x = self.max_pool(ax_x)
        ax_x = F.relu(self.ax_stem_bn2(self.ax_dropout2d2(self.ax_stem2(ax_x))))
        ax_x = self.max_pool(ax_x)
        ax_x = F.relu(self.ax_stem_bn3(self.ax_stem3(ax_x)))
        ax_x = F.relu(self.ax_conv_bn(self.ax_conv(ax_x)))
        ax_x = self.max_pool(ax_x)
        self.feature_maps = ax_x # Store feature maps from axial branch for Grad-CAM
        ax_x = F.relu(self.ax_dw(ax_x))
        ax_x = torch.flatten(ax_x, 1)
        ax_x = F.relu(self.fc_ax(ax_x))

        # Concatenate features from both branches
        x = torch.cat((ax_x, cor_x), dim=1)

        # VAlogmar input is available but not used in the final concatenation in the original forward pass.
        # If it should be used, it needs to be explicitly concatenated.
        # Example: x = torch.cat((ax_x, cor_x, va_logmar_input), dim=1)
        # In that case, fc_final input features (20) would also need to be adjusted.

        return self.fc_final(x)