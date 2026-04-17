import torch
import torch.nn as nn
import torch.nn.functional as F

class axCNN_Model(nn.Module):
    def __init__(self, mode, multiclass=False):
        super(axCNN_Model, self).__init__()
        self.mode = mode
        self.multiclass = multiclass

        self.filters = 32

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()
        self._build_stem()
        self._build_depthwise()
        self._build_final()

    def _build_stem(self):
        #self.ax_1d = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0)
        #self.ax_1d_bn = nn.BatchNorm2d(32)
        stage1 = 128
        stage2 = 64
        stage3 = 32
        self.ax_stem1 = nn.Conv2d(30, stage1, kernel_size=3, stride=1, padding=1)
        self.ax_stem_bn1 = nn.BatchNorm2d(stage1)
        self.Dropout2d1 = nn.Dropout2d(0.5)

        self.ax_stem2 = nn.Conv2d(stage1, stage2, kernel_size=3, stride=1, padding=1)
        self.ax_stem_bn2 = nn.BatchNorm2d(stage2)
        self.Dropout2d2 = nn.Dropout2d(0.5)

        self.ax_stem3 = nn.Conv2d(stage2, stage3, kernel_size=3, stride=1, padding=1)
        self.ax_stem_bn3 = nn.BatchNorm2d(stage3)
        self.Dropout2d3 = nn.Dropout2d(0.5)


        self.ax_stem4 = nn.Conv2d(stage3, 64, kernel_size=3, stride=1, padding=1)
        self.ax_stem_bn4 = nn.BatchNorm2d(64)
        self.Dropout2d4 = nn.Dropout2d(0.2)



    def _build_depthwise(self):
        # Similar to the stem, you can convert each layer with DepthwiseConv2D to PyTorch's equivalent.
        # Due to space constraints, I'll just provide an example for one layer:
        self.ax_conv = nn.Conv2d(self.filters, self.filters, kernel_size=3, stride=1, padding=1, groups=self.filters)
        self.ax_conv_bn = nn.BatchNorm2d(self.filters)

        self.ax_dw = nn.Conv2d(in_channels=self.filters, out_channels=self.filters,
                                 kernel_size=(19, 15),
                                 stride=(19, 15),groups=self.filters, bias=False)

        #self.ax_dw = nn.Conv2d(in_channels=self.filters, out_channels=self.filters,kernel_size=(32, 32),stride=(32, 32),groups=self.filters, bias=False)

    def _build_final(self):
        self.fc_ax = nn.Linear(self.filters+1 , 10)

        #self.fc_ax = nn.Linear(self.filters , self.filters//2)
        self.fc1 = nn.Linear(10, 1)
        self.fc2 = nn.Linear(10, 3)

    def forward(self, x,valog, training=True):
        if 'ax' in self.mode:
            ax_x = x

            # print(ax_x.shape,ax_x.dtype)
            #ax_x = F.tanh(self.ax_1d_bn(self.ax_1d(ax_x)))

            ax_x = F.relu(self.ax_stem_bn1(self.Dropout2d1(self.ax_stem1(ax_x))))
            ax_x = self.max_pool(ax_x)

            ax_x = F.relu(self.ax_stem_bn2(self.Dropout2d2(self.ax_stem2(ax_x))))
            ax_x = self.max_pool(ax_x)

            ax_x = F.relu(self.ax_stem_bn3(self.ax_stem3(ax_x)))
            #ax_x = self.max_pool(ax_x)
            # print('feature map shape 0',ax_x.shape)

            #ax_x = F.relu(self.ax_stem_bn4(self.Dropout2d4(self.ax_stem4(ax_x))))


            ax_x = F.relu(self.ax_conv_bn(self.ax_conv(ax_x)))
            # print('feature map shape 1',ax_x.shape)


            ax_x = self.max_pool(ax_x)
            # print(ax_x.shape)

            # print('feature map shape 2',ax_x.shape)
            self.feature_maps = ax_x

            # print(ax_l.shape,ax_r.shape)
            ax_x = F.relu(self.ax_dw(ax_x))



            #ax_x = F.relu(self.ax_dw(ax_x))
            # print(ax_x.shape)

            ax_x = torch.flatten(ax_x,1)
            # print(ax_x.shape)
            #print(ax_x.shape,valog.unsqueeze(1).shape)
            ax_x = torch.cat((ax_x,valog.unsqueeze(1)),dim=1)
            # print(ax_x.shape)
            ax_x = F.relu(self.fc_ax(ax_x))

            if self.mode == 'ax':
                if self.multiclass:
                    return F.relu(self.fc2(ax_x))
                else:
                    # print(ax_x.shape)
                    return (self.fc1(ax_x))