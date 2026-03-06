import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN3Head(nn.Module):
    # 3 heads: segmentation clas & sification & bbox regression
    def __init__(self, in_ch=3, num_classes=10, base=32):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down1 = nn.MaxPool2d(2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(base, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.down2 = nn.MaxPool2d(2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # cls head
        self.cls_fc = nn.Linear(base * 4, num_classes)

        # bbox head (xyxy in [0,1])
        self.box_fc = nn.Linear(base * 4, 4)

        # seg head
        self.seg_up1 = nn.Sequential(
            nn.Conv2d(base * 4, base * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.seg_up2 = nn.Sequential(
            nn.Conv2d(base * 2, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.seg_out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        f1 = self.enc1(x)
        x = self.down1(f1)
        f2 = self.enc2(x)
        x = self.down2(f2)
        f3 = self.enc3(x)

        g = F.adaptive_avg_pool2d(f3, (1, 1)).flatten(1)

        cls_logits = self.cls_fc(g)
        bbox_pred = torch.sigmoid(self.box_fc(g))  # [B,4] in [0,1], xyxy

        s = F.interpolate(f3, scale_factor=2, mode="bilinear", align_corners=False)
        s = self.seg_up1(s)
        s = F.interpolate(s, scale_factor=2, mode="bilinear", align_corners=False)
        s = self.seg_up2(s)
        mask_logits = self.seg_out(s)

        return mask_logits, cls_logits, bbox_pred
