import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate, dropout_rate=0.2):
        super(DenseLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate, dropout_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([DenseLayer(in_channels + i * growth_rate, growth_rate, dropout_rate) 
                                     for i in range(num_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.2):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.dropout(out)
        out = F.avg_pool2d(out, 2)
        return out

class DenseNet121(nn.Module):
    def __init__(self, growth_rate=32, num_classes=1000, dropout_rate=0.2):
        super(DenseNet121, self).__init__()
        
        num_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.dense1 = DenseBlock(num_channels, 6, growth_rate, dropout_rate)
        num_channels += 6 * growth_rate
        self.trans1 = TransitionLayer(num_channels, num_channels // 2, dropout_rate)
        num_channels = num_channels // 2
        
        self.dense2 = DenseBlock(num_channels, 12, growth_rate, dropout_rate)
        num_channels += 12 * growth_rate
        self.trans2 = TransitionLayer(num_channels, num_channels // 2, dropout_rate)
        num_channels = num_channels // 2
        
        self.dense3 = DenseBlock(num_channels, 24, growth_rate, dropout_rate)
        num_channels += 24 * growth_rate
        self.trans3 = TransitionLayer(num_channels, num_channels // 2, dropout_rate)
        num_channels = num_channels // 2
        
        self.dense4 = DenseBlock(num_channels, 16, growth_rate, dropout_rate)
        num_channels += 16 * growth_rate
        
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(num_channels, num_classes)
        
        self._initialize_weights()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.pool(F.relu(self.bn1(out)))
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        out = F.relu(self.bn2(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.head(out)
        return out
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
