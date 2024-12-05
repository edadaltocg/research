import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ProjectionHead(nn.Module):
    def __init__(
        self, in_dim: int, out_dim: int, inner_dim: int, dropout_p: float
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.inner_dim = inner_dim
        self.dropout_p = dropout_p

        self.fc1 = nn.Linear(in_dim, inner_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(inner_dim, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x: Tensor) -> Tensor:
        y = self.fc1(x)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.fc2(y) + x
        y = self.ln(y)
        return y


class CLIP(nn.Module):
    def __init__(
        self,
        vision_encoder: nn.Module,
        text_encoder: nn.Module,
        vision_encoder_hidden_dim: int,
        text_encoder_hidden_dim: int,
        hidden_dim: int = 512,
        proj_bias=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.vision_encoder_hidden_dim = vision_encoder_hidden_dim
        self.text_encoder_hidden_dim = text_encoder_hidden_dim

        self.vision_proj = nn.Linear(
            vision_encoder_hidden_dim, hidden_dim, bias=proj_bias
        )
        self.text_proj = nn.Linear(text_encoder_hidden_dim, hidden_dim, bias=proj_bias)
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def encode_image(self, image: Tensor, normalize=True) -> Tensor:
        x = self.vision_encoder(image)
        x = self.vision_proj(x)
        if normalize:
            x = F.normalize(x, dim=-1)
        return x

    def encode_text(self, text: Tensor, normalize=True) -> Tensor:
        x = self.text_encoder(text)
        x = self.text_proj(x)
        if normalize:
            x = F.normalize(x, dim=-1)
        return x

    def forward(self, image: Tensor, text: Tensor):
        # (b, c, w, h) -> (b, d)
        image_features = self.encode_image(image)
        # (b, s, v) -> (b, d)
        text_features = self.encode_text(text)
        # (b, d) @ (b, d).T -> (b, b)
        logits = image_features.mm(text_features.t()) * self.temperature.exp()
        labels = torch.arange(logits.size(0)).to(logits.device)
        loss_img = F.cross_entropy(logits, labels)
        loss_txt = F.cross_entropy(logits.T, labels)
        # maximising cosine similarity of encodings on the diagonal of the N*N matrix (image,text) pais
        loss = (loss_img + loss_txt) / 2
        return {
            "logits": logits,
            "image_features": image_features,
            "text_features": text_features,
            "loss_img": loss_img,
            "loss_txt": loss_txt,
            "loss": loss,
        }
