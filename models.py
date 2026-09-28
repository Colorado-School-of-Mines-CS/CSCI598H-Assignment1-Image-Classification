"""Classifier architectures used in the CIFAR-10 comparison."""

import torch
from torch import nn


class LinearClassifier(nn.Module):
    """A single linear layer for image classification."""

    def __init__(self, input_dim: int = 3 * 32 * 32, num_classes: int = 10):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [batch_size, num_classes]."""
        # TODO 1: Flatten each image, but keep the batch dimension.
        # TODO 2: Pass the flattened images through self.linear.
        B, C, H, W = images.shape
        images_reshaped = images.reshape(B, C*H*W)
        return self.linear(images_reshaped)


class ThreeLayerClassifier(nn.Module):
    """A three-layer fully connected classifier (three trainable layers)."""

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=512, num_classes=10):
        super().__init__()
        # TODO: Define Linear(input_dim, hidden_dim), Linear(hidden_dim,
        # hidden_dim), and Linear(hidden_dim, num_classes), with ReLU after
        # each of the first two linear layers.
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.activation = nn.ReLU()
        

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [batch_size, num_classes]."""
        # TODO: Flatten each image and apply the three-layer network.
        B, C, H, W = images.shape
        images_reshaped = images.reshape(B, C*H*W)
        x = self.input_layer(images_reshaped)
        x = self.activation(x)
        x = self.hidden_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x


class ConvClassifier(nn.Module):
    """A small convolutional network for 32 x 32 RGB images."""

    def __init__(self, num_classes=10, num_layers=2):
        super().__init__()
        if num_layers not in {2, 5}:
            raise ValueError("ConvClassifier num_layers must be 2 or 5")
        self.num_layers = num_layers
        # TODO: Build num_layers convolutional blocks. Each block contains
        # Conv2d(kernel_size=3, padding=1), ReLU, and MaxPool2d(2).
        # Use output channels [32, 64] for the 2-layer model and
        # [32, 64, 128, 128, 128] for the 5-layer model. After the blocks,
        # flatten and use Linear(flattened_dim, 128), ReLU, and
        # Linear(128, num_classes). For 32x32 inputs, flattened_dim is
        # channels[-1] * (32 // 2**num_layers) ** 2.
        self.activation = nn.ReLU()
        self.layers = [
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            self.activation,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            self.activation,
            nn.MaxPool2d(kernel_size=2),
        ]
        if self.num_layers == 5:
            self.layers += [ 
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
                self.activation,
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                self.activation,
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
                self.activation,
                nn.MaxPool2d(kernel_size=2),
            ]
        # self.l1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # self.l2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # if self.num_layers == 5:
        #     self.l3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        #     self.l4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        #     self.l5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        # self.relu = nn.ReLU()
        # self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.conv_layers = nn.Sequential(*self.layers)
        # pass noise to the conv layers and get the output shape to calculate the flattened dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 32, 32)
            conv_output = self.conv_layers(dummy_input)
            self.flattened_dim = conv_output.numel()
        self.flatten = nn.Flatten()
        
        self.linear1 = nn.Linear(self.flattened_dim, 128)
        self.linear2 = nn.Linear(128, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [batch_size, num_classes]."""
        # TODO: Apply the convolutional network and return logits.
        x = self.conv_layers(images)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class VisionTransformerClassifier(nn.Module):
    """A compact Vision Transformer for 32 x 32 RGB images.

    Required architecture:
    - split each image into non-overlapping 4 x 4 patches;
    - embed every patch into 128 dimensions;
    - prepend one learnable classification token;
    - add learnable positional embeddings;
    - apply either two or five Transformer encoder layers, each with four
      attention heads and a 256-dimensional feed-forward block;
    - classify the final classification-token representation.
    """

    def __init__(
        self,
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=128,
        num_heads=4,
        mlp_dim=256,
        num_layers=2,
        num_classes=10,
        dropout=0.1,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if num_layers not in {2, 5}:
            raise ValueError("VisionTransformerClassifier num_layers must be 2 or 5")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # TODO: Define all components below.
        # 1. Use Conv2d(in_channels, embed_dim, kernel_size=patch_size,
        #    stride=patch_size) as the patch embedding.
        # 2. Define learnable cls_token [1, 1, embed_dim] and positional
        #    embedding [1, num_patches + 1, embed_dim] parameters.
        # 3. Create nn.TransformerEncoderLayer with d_model=embed_dim,
        #    nhead=num_heads, dim_feedforward=mlp_dim, dropout=dropout, and
        #    batch_first=True. Wrap it in nn.TransformerEncoder with
        #    num_layers encoder layers.
        # 4. Add a final LayerNorm and Linear(embed_dim, num_classes) head.
        self.patch_embeding = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.CLS_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        # self.positional_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        #figuring out dimsion was kicking my ass
        pos_embed = self.positional_encoding_2d(
            self.image_size // self.patch_size,
            self.image_size // self.patch_size,
            embed_dim)
        #copied from attention is all you need pytorch
        self.register_buffer('positional_embedding', pos_embed)



        # Dr li wants this by hand implment later 
        # self.transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(
        #         d_model=embed_dim, 
        #         nhead=num_heads, 
        #         dim_feedforward=mlp_dim, 
        #         dropout=dropout, 
        #         batch_first=True
        #     ),
        #     num_layers=num_layers
        # )

        self.transformer = Transformer(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            mlp_dim=mlp_dim, 
            dropout=dropout, 
            num_layers=num_layers
        )


        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.output = nn.Linear(embed_dim, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return logits with shape [batch_size, num_classes]."""
        # TODO: Check the input spatial size, patch-embed the images, flatten
        # the patch grid into a token sequence, prepend a copy of cls_token for
        # each item, add positional embeddings, run the encoder, and
        # classify the normalized class token. Return logits, not probabilities.
        x = self.patch_embeding(images)
        B, D_embed, H_patch, W_patch = x.shape
        #turn into a sequence of tokens 
        x = x.flatten(2).transpose(1, 2)  # [B, num_tokens/pathces, embed_dim]
        x = torch.cat([self.CLS_token.expand(B, -1, -1), x], dim=1)  # [B, num_patches + 1, embed_dim]

        x += self.positional_embedding
        x = self.transformer(x)
        cls_token = x[:, 0, :]
        cls_token = self.layer_norm(cls_token)  # [B, embed_dim]
        logits = self.output(cls_token)  # [B, num_classes]
        return logits
     

    def positional_encoding_1d(self, seq_length, d_model):
        '''
            sin(pos / 10000^(2i pos/d_model))
            cos (pos / 10000^(2i pos/d_model)) 

            in logarithm we have
            log(10000^(2i pos/d_model)) = 2 pos/d_model * log(10000) = exp(2* 4 * pos/d_model * log(10))
        '''
        pos = torch.arange(seq_length).unsqueeze(1) # 8 tokens
        idx = torch.arange(d_model // 2).unsqueeze(0)
        powers =  2 * idx / d_model
        denominator = torch.exp(powers * torch.log(torch.tensor(10_000.0)))
        return torch.cat([torch.sin(pos / denominator), torch.cos(pos / denominator)], dim=1)

    def positional_encoding_2d(self, seq_length_x, seq_length_y, d_model):
        # inspired by nvigational world model code base
        d_k = d_model//2
        #shapes: [seq_lenght,dk]
        pos_x = self.positional_encoding_1d(seq_length_x, d_k)
        pos_y = self.positional_encoding_1d(seq_length_y, d_k)
        # for each col and row i need to repeat encoding
        pos_x_table = pos_x.repeat(seq_length_y, 1) #y coord repeat for each x cord
        pos_y_table = pos_y.repeat_interleave(seq_length_x, dim=0) # x coord repeat for each y coord

        positional_encoding=  torch.cat([pos_x_table, pos_y_table], dim=1)
        cls_pos = torch.zeros(1, d_model, device=positional_encoding.device)
        return torch.cat([cls_pos, positional_encoding], dim=0).unsqueeze(0)

class MHA_Attention(nn.Module):
    def __init__(self, 
                        embed_dim, 
                        num_heads, 
                        mlp_dim,
                        dropout=0.1, 
                        #num_layers=2
                    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        # self.num_layers = num_layers
        self.d_k = embed_dim // num_heads

        self.Query = nn.Linear(embed_dim, embed_dim)
        self.Key = nn.Linear(embed_dim, embed_dim)
        self.Value = nn.Linear(embed_dim, embed_dim)
        # self.mlp = nn.Sequential(
        #     nn.Linear(embed_dim, mlp_dim),
        #     nn.ReLU(),
        #     nn.Linear(mlp_dim, embed_dim)
        # )

        #MLP layer
    def attention(self, Q, K, V):
        inner_prod = torch.matmul(Q, K.transpose(-2, -1))
        inner_prod /= (self.d_k ** 0.5)
        attn = torch.softmax(inner_prod, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, V)
    def forward(self, x):
        #self attention so all inputs are the same
        Q = self.Query(x)
        K = self.Key(x)
        V = self.Value(x)
        #chunking for multi head attnetion
        Q = Q.chunk(self.num_heads, dim=-1)
        K = K.chunk(self.num_heads, dim=-1)
        V = V.chunk(self.num_heads, dim=-1)
        out = [self.attention(q, k, v) for q, k, v in zip(Q, K, V)]
        out = torch.cat(out, dim=-1)

        #MLP layer 
        # out = self.mlp(out)
        return out

class TransformerEncoderLayer(nn.Module):
    def __init__(self, 
                        embed_dim, 
                        num_heads, 
                        mlp_dim,
                        dropout,
                        # num_layers
                        ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.mha = MHA_Attention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            mlp_dim=mlp_dim, 
            dropout=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
    def forward(self, x):
        x  = x + self.mha(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x
        # input_x = x
        # x = self.layer_norm(x)
        # for attn_layer in self.layer_stack:
        #     x = input_x + attn_layer(x)
        #     input_x = x

        # return x
class Transformer(nn.Module):
    def __init__(self, 
                        embed_dim, 
                        num_heads, 
                        mlp_dim,
                        dropout,
                        num_layers
                        ):
        super().__init__()
        self.layer_stack =  nn.Sequential(*[
            TransformerEncoderLayer(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                mlp_dim=mlp_dim, 
                dropout=dropout)
            for _ in range(num_layers)
        ])
    def forward(self, x):
        return self.layer_stack(x)

def build_model(name: str, num_classes=10, num_layers=None) -> nn.Module:
    """Create one of the assignment's classifier architectures."""
    normalized = name.lower().replace("-", "_")
    if normalized == "linear":
        return LinearClassifier(num_classes=num_classes)
    if normalized in {"three_layer", "mlp"}:
        return ThreeLayerClassifier(num_classes=num_classes)
    if normalized in {"conv", "cnn"}:
        return ConvClassifier(num_classes=num_classes, num_layers=num_layers or 2)
    if normalized in {"vit", "vision_transformer"}:
        return VisionTransformerClassifier(
            num_classes=num_classes, num_layers=num_layers or 2
        )
    raise ValueError(
        f"Unknown model {name!r}; choose linear, three_layer, conv, or vit."
    )
