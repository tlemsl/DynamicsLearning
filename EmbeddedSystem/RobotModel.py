import torch
import torch.nn as nn
class RobotDynamicsModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(RobotDynamicsModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 2 * input_size, bias=True)
        self.fc2 = nn.Linear(2 * input_size, 4 * input_size, bias=True)
        self.fc3 = nn.Linear(4 * input_size, 6 * input_size, bias=True)
        self.fc4 = nn.Linear(6 * input_size, 2 * input_size, bias=True)
        self.output_layer = nn.Linear(2 * input_size, output_size, bias=True)  # Output size is |x| = 3

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.output_layer(x)
        return x


import torch
import torch.nn as nn

class TransformerDecoderModel2D(nn.Module):
    def __init__(self, input_size, output_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerDecoderModel2D, self).__init__()
        
        # Project input from input_size (100) to d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output layer to map d_model back to output_size (3)
        self.output_projection = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, feature_dim)
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        batch_size, feature_dim = x.shape

        # Project input to d_model
        x = self.input_projection(x)  # Shape: (batch_size, d_model)

        # Treat features as sequence for Transformer (seq_len=1, batch_size, d_model)
        x = x.unsqueeze(0)  # Add seq_len dimension: (1, batch_size, d_model)

        # Dummy memory input for standalone decoder
        memory = torch.zeros_like(x)

        # Pass through Transformer Decoder
        x = self.transformer_decoder(x, memory)  # Shape: (seq_len=1, batch_size, d_model)

        # Remove seq_len dimension
        x = x.squeeze(0)  # Shape: (batch_size, d_model)

        # Project to output size
        x = self.output_projection(x)  # Shape: (batch_size, output_size)
        return x


import math

class LoRALayer(nn.Module):
    def __init__(self, original_layer, r=4, alpha=8):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # LoRA 레이어 정의
        self.lora_a = nn.Linear(original_layer.in_features, r, bias=False)
        self.lora_b = nn.Linear(r, original_layer.out_features, bias=False)

        # LoRA 레이어를 서브모듈로 등록
        self.add_module("lora_a", self.lora_a)
        self.add_module("lora_b", self.lora_b)

        # Initialize LoRA layers
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x):
        # 기존 출력 + LoRA의 출력
        return self.original_layer(x) + self.scaling * self.lora_b(self.lora_a(x))


class TransformerDecoderModelWithLoRA(TransformerDecoderModel2D):
    def __init__(self, *args, lora_r=4, lora_alpha=8, **kwargs):
        super(TransformerDecoderModelWithLoRA, self).__init__(*args, **kwargs)

        # input_projection에 LoRA 적용
        self.input_projection = LoRALayer(self.input_projection, r=lora_r, alpha=lora_alpha)
        # output_projection에 LoRA 적용
        self.output_projection = LoRALayer(self.output_projection, r=lora_r, alpha=lora_alpha)
