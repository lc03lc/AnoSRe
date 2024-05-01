import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models
from torchvision.models import ViT_B_16_Weights
from transformers import AutoModelForSequenceClassification
from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection


class AnoSRe(nn.Module):
    def __init__(self, n_category, vision_model="vit_b_16", text_model="roberta_base",
                 structured_data_model="FD", vision_encoder_clip="vision_clip",
                 text_encoder_clip="text_clip"):
        super().__init__()
        model_dic = {
            "vit_b_16": Vit_b_16,
            "roberta_base": Roberta_base,
            "VecBlock": VecBlock,
            "vision_clip": ImageClipModel,
            "text_clip": TextClipModel
        }

        self.vision_encoder = model_dic[vision_model]()
        self.text_encoder = model_dic[text_model]()
        self.data_encoder = model_dic[structured_data_model]()
        self.vision_encoder_clip = model_dic[vision_encoder_clip]()
        self.text_encoder_clip = model_dic[text_encoder_clip]()

        self.flatten = nn.Flatten()
        self.att = nn.TransformerEncoderLayer(d_model=100, nhead=2)
        self.fc1 = nn.Linear(600, 50)
        self.bn1 = nn.BatchNorm1d(num_features=50)
        self.ln1 = nn.LayerNorm(normalized_shape=50)
        self.fc2 = nn.Linear(50, n_category)

    def forward(self, img, img_clip, text, text_clip, dis, score):
        VE_i = self.vision_encoder(img)
        TRE_t = self.text_encoder(text)
        FD = self.data_encoder(torch.cat([dis, score.unsqueeze(1)], dim=1))
        FI = self.vision_encoder_clip(img_clip)
        FT = self.text_encoder_clip(text_clip)

        x = torch.stack([VE_i, TRE_t, FD, FI, FT], dim=1)
        x = self.att(x)
        x = self.flatten(x)
        x = torch.cat([x, VE_i], dim=1)
        x = F.relu(self.ln1(self.fc1(x)))
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class Vit_b_16(nn.Module):
    def __init__(self, pretrained=True, n_class=100):
        super().__init__()

        weight = ViT_B_16_Weights.DEFAULT if pretrained else None
        self.base_model_vit = models.vit_b_16(weights=weight)
        self.base_model_vit.heads.head = nn.Linear(self.base_model_vit.heads.head.in_features, n_class)

    def forward(self, x):
        x = self.base_model_vit(x)
        return x


class Roberta_base(nn.Module):
    def __init__(self, n_class=100):
        super().__init__()

        self.text_model = "roberta-base"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.text_model, num_labels=n_class)

    def forward(self, x):
        x = self.model(x)
        return x.logits


class VecBlock(nn.Module):
    def __init__(self, in_features=21, n_class=100):
        super().__init__()

        self.fc1 = nn.Linear(in_features, 120)
        self.fc2 = nn.Linear(120, n_class)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class ImageClipModel(nn.Module):
    def __init__(self, n_class=100):
        super().__init__()
        self.clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, n_class)

    def forward(self, x):
        x = self.clip(x).image_embeds
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class TextClipModel(nn.Module):
    def __init__(self, n_class=100):
        super().__init__()
        self.clip = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, n_class)

    def forward(self, x):
        x = self.clip(x).text_embeds
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x
