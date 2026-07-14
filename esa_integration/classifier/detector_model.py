"""
models.py — P1 검출 사다리 (docs/06 §1–2, §5.1)

  B1 PlainCNN : conv 백본 + GAP 풀링 — attention 기여 측정용 베이스라인
  B2 CNNAttn  : 동일 백본 + Bahdanau temporal attention (방식 A 검출기)
  B3 ViT      : 8×8 패치 + Pre-LN Transformer ×4 (방식 B 검출기)

입력: (B, 1, 64, 216) 로그멜 (dataset.py, 윈도우별 정규화)
출력: (B, 3) logits — {siren, horn, noise} = dataset.LABEL_IDX 순서
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_CLASSES = 3


class _ConvBackbone(nn.Module):
    """Conv 3×3 (32→64→128) · BN · ReLU · MaxPool2 — Salamon & Bello 2017 스타일.
    (B,1,64,216) → (B,128,8,27)"""

    def __init__(self):
        super().__init__()
        chs = [1, 32, 64, 128]
        layers = []
        for i in range(3):
            layers += [
                nn.Conv2d(chs[i], chs[i + 1], 3, padding=1, bias=False),
                nn.BatchNorm2d(chs[i + 1]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _Classifier(nn.Module):
    """Dropout → Linear(128→64) → ReLU → Linear(64→3) (docs/06 §1.1)"""

    def __init__(self, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(p_drop), nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Linear(64, N_CLASSES)
        )

    def forward(self, x):
        return self.net(x)


class PlainCNN(nn.Module):
    """B1 — 백본 + 주파수 평균 + 시간 GAP. B2와의 차이 = attention의 기여."""

    def __init__(self):
        super().__init__()
        self.backbone = _ConvBackbone()
        self.cls = _Classifier()

    def forward(self, x):
        h = self.backbone(x).mean(dim=2)      # (B,128,27) — 주파수축 평균
        return self.cls(h.mean(dim=2))        # 시간 GAP → (B,3)


class CNNAttn(nn.Module):
    """B2 — temporal attention 풀링 (Bahdanau additive scoring, Kong 2018 풀링).
    증거가 시간상 희소할 때(GAP 대비) 우월 — 슬라이딩 청크에서 사이렌이 잘리는 상황."""

    def __init__(self):
        super().__init__()
        self.backbone = _ConvBackbone()
        self.score = nn.Sequential(nn.Linear(128, 64), nn.Tanh(), nn.Linear(64, 1))
        self.cls = _Classifier()

    def features(self, x):
        """검출 head 앞 context 벡터 (B,128) — 통합모델의 속도/차종 head 공유 입력."""
        h = self.backbone(x).mean(dim=2).permute(0, 2, 1)   # (B,27,128)
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)  # (B,27) 시간축 가중
        return (h * w.unsqueeze(-1)).sum(dim=1)              # (B,128)

    def forward(self, x, return_attn: bool = False):
        h = self.backbone(x).mean(dim=2).permute(0, 2, 1)
        w = torch.softmax(self.score(h).squeeze(-1), dim=1)
        ctx = (h * w.unsqueeze(-1)).sum(dim=1)
        out = self.cls(ctx)
        return (out, w) if return_attn else out


class ViT(nn.Module):
    """B3 — AST 스타일 멜 패치 ViT (docs/06 §2.1).
    8×8 패치(겹침 없음) → 8×27=216 토큰 + CLS, Pre-LN encoder ×4 (dim 128, 4 heads, MLP×4)."""

    def __init__(self, depth=4, dim=128, heads=4, p_drop=0.1):
        super().__init__()
        self.patch = nn.Conv2d(1, dim, kernel_size=8, stride=8)          # (B,128,8,27)
        n_tokens = (64 // 8) * (216 // 8) + 1                            # 217
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4, dropout=p_drop,
            activation="gelu", batch_first=True, norm_first=True,        # Pre-LN (Xiong 2020)
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64, N_CLASSES))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def features(self, x):
        """검출 head 앞 CLS 특징 (B,128) — 통합모델 공유 입력."""
        h = self.patch(x).flatten(2).permute(0, 2, 1)
        cls = self.cls_token.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1) + self.pos
        h = self.encoder(h)
        return self.norm(h)[:, 0]                                       # (B,128)

    def forward(self, x):
        return self.head(self.features(x))


MODELS = {"cnn": PlainCNN, "cnn_attn": CNNAttn, "vit": ViT}


def build(name: str) -> nn.Module:
    return MODELS[name]()


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    x = torch.randn(2, 1, 64, 216)
    for name in MODELS:
        m = build(name)
        y = m(x)
        print(f"{name:8s} params {n_params(m)/1e6:.3f}M  out {tuple(y.shape)}")
    _, w = CNNAttn()(x, return_attn=True)
    print(f"cnn_attn attention weights {tuple(w.shape)} (합 {w.sum(dim=1).tolist()})")
