import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Args:
            in_channels (int): 輸入通道數。
            out_channels (int): 輸出通道數。
        Description:
            實作 2015 Unet 論文原版的【Blue Arrow】-> (Conv 3x3, ReLU)*2 。
            卷積核大小為 3x3，且不使用 padding，因此輸出特徵圖的空間尺寸會比輸入小 2 像素。
        """

        # 使用 inplace=True 以節省內存。(直接在原始的 Tensor 記憶體位置上進行計算並覆寫結果，而不會額外配置一塊新的記憶體來存放輸出。)
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 輸入特徵圖，形狀為 (B, in_channels, H, W)。
        Returns:
            (torch.Tensor): 輸出特徵圖，形狀為 (B, out_channels, H-4, W-4)。
        Description:
            將輸入特徵圖通過兩個卷積層和 ReLU 激活函數的序列進行處理，並返回結果。
        """

        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, feature_channels=[64, 128, 256, 512]):
        """
        Args:
            in_channels (int): 輸入圖像的通道數 (預設為 3, RGB 圖像)。
            out_channels (int): 輸出圖像的通道數 (預設為 1, 單通道二值圖像)。
            feature_channels (list): 各層卷積的通道數列表 (預設為 [64, 128, 256, 512])。
        Description:
            實作 2015 Unet 論文原S版的【架構】：
            - Encoder (U 左側，下采樣用):
                - 4 層 DoubleConv ，每層卷積後接一個 2x2 的 Max Pooling 層，通道數分別為 64、128、256、512。
            - Decoder (U 右側，上採樣用): 
                - 4 層 DoubleConv ，每層卷積前接一個 2x2 的 Up-conv 層，通道數分別為 512、256、128、64。
            - Skip Connections (U 中間，跳躍連接用): 
                - 在 Decoder 的每一層卷積前，將對應的 Encoder 層的輸出與上採樣後的特徵圖進行 concatenate ，以保留高分辨率的特徵信息。
            - Bottleneck (U 下方，連接用):
                - 在 Encoder 的最後一層和 Decoder 的第一層之間，加入一個卷積層，通道數為 1024 ，作為 Encoder 和 Decoder 之間的橋樑。
            - 最終輸出層:
                - 將通道數轉換為 out_channels。
        """

        super(UNet, self).__init__()

        # Encoder
        self.downs = nn.ModuleList()  # 存儲 Encoder 層 Module 用。
        for feature in feature_channels:
            self.downs.append(DoubleConv(in_channels, feature)) # 添加 DoubleConv 層 【Blue Arrow * 2】。
            in_channels = feature                               # 更新 in_channels 為當前層的輸出通道數。

        # Bottleneck
        self.bottleneck = DoubleConv(feature_channels[-1], feature_channels[-1] * 2)    # 將 Encoder 的最後一層通道數翻倍。

        # Decoder
        self.ups = nn.ModuleList()    # 存儲 Decoder 層 Module 用。
        for feature in reversed(feature_channels):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))  # 添加 Up-conv 層 【Green Arrow】。
            self.ups.append(DoubleConv(feature * 2, feature))                                   # 添加 DoubleConv 層 【Blue Arrow * 2】。

        # 最終輸出層
        self.final_conv = nn.Conv2d(feature_channels[0], out_channels, kernel_size=1)  # 使用 1x1 卷積將通道數轉換為 out_channels。
    
    def center_crop(self, x, target_height, target_width):
        """
        Args:
            x (torch.Tensor): 輸入特徵圖，形狀為 (B, C, H, W)。
            target_height (int): 目標高度。
            target_width (int): 目標寬度。
        Returns:
            (torch.Tensor): 裁剪後的特徵圖，形狀為 (B, C, target_height, target_width)。
        Description:
            由於 UNet 的設計會導致 Decoder 層的特徵圖大小與對應的 Encoder 層的特徵圖大小不匹配，因此需要進行中心裁剪以對齊大小。
        """

        _, _, h, w = x.size()
        start_h = (h - target_height) // 2
        start_w = (w - target_width) // 2
        return x[:, :, start_h:start_h + target_height, start_w:start_w + target_width]
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 輸入圖像，形狀為 (B, C, H, W)。
        Returns:
            (torch.Tensor): 輸出圖像，形狀為 (B, out_channels, H, W)。
        Description:
            實現 UNet 的前向傳播過程，包含 Encoder、Bottleneck、Decoder 和 Skip Connections 的操作。
        """

        skip_connections = []       # 存儲 Encoder 層的輸出，之後用在 Decoder。

        # Encoder
        for down in self.downs:
            x = down(x)                                     # 通過 DoubleConv 層。      【Blue Arrow * 2】
            skip_connections.append(x)                      # 保存當前層的輸出。         【Gray Arrow】
            x = nn.MaxPool2d(kernel_size=2, stride=2)(x)    # Max Pooling。             【Red Arrow】

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections.reverse()  # 反轉使其與 Decoder 層對應。
        for i in range(0, len(self.ups), 2):            # 因為上面設計兩個元素為一組 (Up-conv + DoubleConv)，所以要兩個兩個跳。
            x = self.ups[i](x)                          # Up-conv。                     【Green Arrow】
            skip_connection = skip_connections[i // 2]  # 獲取對應的 skip_connections。  【White Block】
            if x.shape != skip_connection.shape:        # 如果 Up-conv 後的特徵圖與 skip_connection 的大小不匹配，則進行中心裁剪。
                skip_connection = self.center_crop(skip_connection, x.shape[2], x.shape[3])  # 中心裁剪 (H, W)。
            concat_skip = torch.cat((skip_connection, x), dim=1)  # 將 skip_connection 與 Up-conv 後的特徵圖在通道維度 (C, dim=1) 上進行 concatenate。
            x = self.ups[i + 1](concat_skip)            # 通過 DoubleConv 層。           【Blue Arrow * 2】

        # 最終輸出層。
        return self.final_conv(x)
    
if __name__ == "__main__":
    print("=== Start 【UNet】 Forward Pass Test ===")

    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(1, 3, 572, 572)  # 模擬一個輸入圖像，形狀為 (B, C, H, W)。
    output_x = model(x)

    print(f"[Original Paper Size Test] Input: {x.shape} -> Output: {output_x.shape}") # 預期輸出圖像，形狀為: (1, 1, 388, 388).