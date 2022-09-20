import torch.nn as nn


class SimpleDNN(nn.Module):
    def __init__(self, ):
        """DNNの層を定義
        """
        self.net = nn.Sequential(
            nn.LazyConv2d(out_channels=16, kernel_size=5, padding=2, bias=False),
            nn.LazyBatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=16, kernel_size=5, padding=2, bias=False),
            nn.LazyBatchNorm2d(num_features=16),
            nn.ReLU()
            nn.AdaptiveAvgPool2d(1)
        )
        self.classifier = nn.LazyLinear(out_features=2)
        
    
    def forward(self, x):
        """DNNの入力から出力までの計算
        Args:
            x: torch.Tensor whose size of
               (batch size, # of channels, # of freq. bins, # of time frames)
        Return:
            y: torch.Tensor whose size of
               (batch size, # of classes)
        """
        x = self.net(x)
        y = self.classifier(x.view(x.size(0), -1))
        return y
