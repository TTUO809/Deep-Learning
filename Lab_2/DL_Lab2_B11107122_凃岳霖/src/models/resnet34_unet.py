import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x  
        x = self.sa(x) * x  
        return x

class BasicBlock(nn.Module):
    '''
    ResNet34 + UNet 的基本卷積塊。包含兩層 3x3 卷積 、 Batch Normalization，以及 ReLU 激活函數，並且具有 skip connection。
    '''
    def __init__(self, in_channels, out_channels, stride=1):
        '''
        Args:
            in_channels (int): 輸入通道數。
            out_channels (int): 輸出通道數。
            stride (int): 卷積層的步幅，默認為 1。
        '''
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)  # C(in_channels->out_channels) , 3X3. (stride=stride, padding=1 for same size if stride=1, or half size if stride=2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)   # 沒有需要學習的參數，直接使用 inplace=True 來節省內存，並且只需要宣告一個。
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)      # C(out_channels->out_channels) , 3X3. (stride=1, padding=1 for same size)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # ResNet 的 skip connection。如果步幅不為 1 或者通道數不同，為了要保證殘差相加時形狀一致，則需要使用一個卷積層來調整尺寸。
        # stride != 1：代表主幹道會把圖片的長寬縮小（size 128^2 -> 64^2）。
        # in_channels != out_channels：代表主幹道會把圖片的厚度變厚（C 64 -> 128）。
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): 輸入特徵圖，形狀為 (N, in_channels, H, W)。
        Returns:
            (torch.Tensor): 輸出特徵圖，形狀為 (N, out_channels, H', W')，其中 H' 和 W' 取決於 stride 的值。
        '''
        shortcut = self.downsample(x)

        # 殘差塊: Conv -> BN -> ReLU -> Conv -> BN -> [Add(Shortcut)] -> ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += shortcut
        out = self.relu(out)
        return out
    
class DecoderBlock(nn.Module):
    '''
    ResNet34 + UNet 的 Decoder 塊。包含一個 2x2 的 Up-conv 和一個 BasicBlock 卷積塊，並且在上採樣後與對應的 Encoder 特徵圖進行 concatenate。
    '''
    def __init__(self, in_channels, out_channels=32):
        '''
        Args:
            in_channels (int): 輸入通道數。
            out_channels (int): 輸出通道數。

        '''
        super(DecoderBlock, self).__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_channels)

    def forward(self, x, skip_x):
        '''
        Args:
            x (torch.Tensor): 從上一層 Decoder 傳來的特徵圖，形狀為 (N, in_channels, H, W)。
            skip_x (torch.Tensor): 從 Encoder 傳來的跳躍連接特徵圖，形狀為 (N, skip_channels, H*2, W*2)。
        Returns:
            (torch.Tensor): 輸出特徵圖，形狀為 (N, out_channels, H*2, W*2)。
        '''
        x = self.upsample(x)    # 放大
        x = self.conv(x)        # 濃縮成 32 通道 (藍色區塊)
        x = self.cbam(x)        # 聚焦注意力
        
        # 如果有 Skip Connection，拼接在 32 通道之後
        if skip_x is not None:
            if x.shape[2:] != skip_x.shape[2:]:
                x = F.interpolate(x, size=(skip_x.shape[2], skip_x.shape[3]), mode='bilinear', align_corners=False)
            x = torch.cat([x, skip_x], dim=1) 
        return x
    
class ResNet34UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        '''
        Args:
            in_channels (int): 輸入圖像的通道數。
            out_channels (int): 輸出圖像的通道數。
        '''
        super(ResNet34UNet, self).__init__()
        
        # =============== Encoder =============== 
        # (conv1) [Conv] -> [BN] -> [ReLU]
        # [conv1 / Conv] C(3->64) , 7X7 , stride=2. (padding=3 for half size [224^2->112^2])
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # (conv2_x) [MaxPool] -> [BasicBlock * 3]
        # [conv2_x / MaxPool] 3X3 , stride=2. (padding=1 for half size [112^2->56^2], C(64->64) just for downsampling)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # [conv2_x / BasicBlock * 3] C(64->64)  , 3X3. (stride=1, padding=1 for same size [56^2 -> 56^2])
        self.layer1 = self._make_layer(64, 64, 3, stride=1) 

        # [conv3_x / BasicBlock * 4] C(64->128) , 3X3. (stride=2, padding=1 for half size [56^2 -> 28^2])
        self.layer2 = self._make_layer(64, 128, 4, stride=2)

        # [conv4_x / BasicBlock * 6] C(128->256) , 3X3. (stride=2, padding=1 for half size [28^2 -> 14^2])
        self.layer3 = self._make_layer(128, 256, 6, stride=2)

        # [conv5_x / BasicBlock * 3] C(256->512) , 3X3. (stride=2, padding=1 for half size [14^2 -> 7^2])
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # =============== Bottleneck ===============
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # =============== Decoder ===============
        # Bottleneck (256) + layer4 (512) = 768 -> dec1 (32)
        self.dec1 = DecoderBlock(in_channels=768, out_channels=32)

        # dec1 (32) + layer3 (256) = 288 -> dec2 (32)
        self.dec2 = DecoderBlock(in_channels=288, out_channels=32)

        # dec2 (32) + layer2 (128) = 160 -> dec3 (32)
        self.dec3 = DecoderBlock(in_channels=160, out_channels=32)

        # dec3 (32) + layer1 (64) = 96 -> dec4 (32)
        self.dec4 = DecoderBlock(in_channels=96, out_channels=32)
        
        # =============== Final ===============
        # dec4 (32) + None (0) = 32 -> dec5 (32)
        self.dec5 = DecoderBlock(in_channels=32, out_channels=32)  

        # 最終輸出層，將通道數從 32 轉換為 out_channels。
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        '''
        Args:
            in_channels (int): 輸入通道數。
            out_channels (int): 輸出通道數。
            blocks (int): 區塊數量。
            stride (int): 步幅。
        Returns:
            (nn.Sequential): 包含指定數量 BasicBlock 的 Sequential 模組。
        *** 用來構建 ResNet34 中的每一層卷積塊 (conv2_x, conv3_x, conv4_x, conv5_x)。
        '''
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): 輸入圖像，形狀為 (N, in_channels, H, W)。
        Returns:
            (torch.Tensor): 輸出圖像，形狀為 (N, out_channels, H, W)。
        '''
        orig_size = x.shape[2:]

        # --- Encoder ---
        c0 = self.relu(self.bn1(self.conv1(x)))
        pooled = self.maxpool(c0)

        c1 = self.layer1(pooled)    # Skip_1: 56x56, C=64
        c2 = self.layer2(c1)        # Skip_2: 28x28, C=128
        c3 = self.layer3(c2)        # Skip_3: 14x14, C=256
        c4 = self.layer4(c3)        # Skip_4:  7x7 , C=512

        # --- Bottleneck ---
        btnk = self.bottleneck(c4)              # btnk: 7x7, C=256
        btnk_c4 = torch.cat([btnk, c4], dim=1)  # btnk_c4: 7x7, C=768 (btnk + Skip_4 = 256 + 512)

        # --- Decoder ---
        d1 = self.dec1(btnk_c4, skip_x=c3)      # d1: 14x14,   C=32 (32 + Skip_3 = 32 + 256)
        d2 = self.dec2(d1,      skip_x=c2)      # d2: 28x28,   C=32 (32 + Skip_2 = 32 + 128)
        d3 = self.dec3(d2,      skip_x=c1)      # d3: 56x56,   C=32 (32 + Skip_1 = 32 + 64)
        d4 = self.dec4(d3,      skip_x=None)    # d4: 112x112, C=32 (32)
        d5 = self.dec5(d4,      skip_x=None)    # d5: 224x224, C=32 (32)
        
        out = self.final_conv(d5)

        return out

if __name__ == "__main__":
    print("=== 測試 ResNet34 + UNet 模型 ===")
    
    # 模擬 DataLoader 吐出圖片
    x = torch.randn((4, 3, 572, 572))

    # 測試 ResNet34_UNet 模型的前向傳播。
    model = ResNet34UNet(in_channels=3, out_channels=1)
    output_x = model(x)
    
    print(f"輸入: {x.shape} -> 輸出: {output_x.shape}")
    assert output_x.shape == (4, 1, 572, 572), "錯誤：輸出形狀不符合 (Batch, 1, 572, 572)！"