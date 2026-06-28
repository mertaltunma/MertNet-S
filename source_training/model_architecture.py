import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.pool(x)
        scale = self.fc(scale)
        return x * scale


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_se=True, dropout=0.0):
        super().__init__()

        self.use_residual = (stride == 1 and in_channels == out_channels)

        self.depthwise = ConvBNAct(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            groups=in_channels
        )

        self.pointwise = ConvBNAct(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1
        )

        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.se(out)
        out = self.dropout(out)

        if self.use_residual:
            out = out + x

        return out


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        branch_channels = out_channels // 2

        self.branch_3x3 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, kernel_size=3, stride=stride, groups=1),
            ConvBNAct(branch_channels, branch_channels, kernel_size=3, stride=1, groups=branch_channels)
        )

        self.branch_5x5 = nn.Sequential(
            ConvBNAct(in_channels, branch_channels, kernel_size=5, stride=stride, groups=1),
            ConvBNAct(branch_channels, branch_channels, kernel_size=3, stride=1, groups=branch_channels)
        )

        self.fuse = ConvBNAct(out_channels, out_channels, kernel_size=1)
        self.se = SqueezeExcitation(out_channels)

    def forward(self, x):
        out_3x3 = self.branch_3x3(x)
        out_5x5 = self.branch_5x5(x)

        out = torch.cat([out_3x3, out_5x5], dim=1)
        out = self.fuse(out)
        out = self.se(out)

        return out


class MertNetS(nn.Module):
    def __init__(self, num_classes=1000, dropout=0.3):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNAct(3, 32, kernel_size=3, stride=2),
            ConvBNAct(32, 48, kernel_size=3, stride=1)
        )

        self.stage1 = nn.Sequential(
            DepthwiseSeparableBlock(48, 64, stride=2, use_se=False),
            DepthwiseSeparableBlock(64, 64, stride=1, use_se=True)
        )

        self.stage2 = nn.Sequential(
            DepthwiseSeparableBlock(64, 128, stride=2, use_se=True),
            DepthwiseSeparableBlock(128, 128, stride=1, use_se=True, dropout=0.05)
        )

        self.stage3 = nn.Sequential(
            MultiScaleBlock(128, 192, stride=2),
            DepthwiseSeparableBlock(192, 192, stride=1, use_se=True, dropout=0.05)
        )

        self.stage4 = nn.Sequential(
            DepthwiseSeparableBlock(192, 256, stride=2, use_se=True, dropout=0.10),
            DepthwiseSeparableBlock(256, 256, stride=1, use_se=True, dropout=0.10)
        )

        self.stage5 = nn.Sequential(
            MultiScaleBlock(256, 384, stride=2),
            DepthwiseSeparableBlock(384, 384, stride=1, use_se=True, dropout=0.10)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(384, num_classes)
        )

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    model = MertNetS(num_classes=1000, dropout=0.3)

    dummy_input = torch.randn(1, 3, 160, 160)
    output = model(dummy_input)
    features = model.forward_features(dummy_input)

    total_params, trainable_params = count_parameters(model)

    print("=" * 70)
    print("MertNet-S Mimari Kontrolü")
    print("=" * 70)
    print(model)
    print("=" * 70)
    print(f"Giriş boyutu: {tuple(dummy_input.shape)}")
    print(f"Çıkış boyutu: {tuple(output.shape)}")
    print(f"Özellik vektörü boyutu: {tuple(features.shape)}")
    print(f"Toplam parametre sayısı: {total_params:,}")
    print(f"Eğitilebilir parametre sayısı: {trainable_params:,}")