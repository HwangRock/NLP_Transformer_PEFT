import torch
import torch.nn as nn
from attention import MultiHeadAttention
from layers import Mlp
from embedding import calcul_location
from Transformer.mask import padding_mask


# Encoder에서 사용할 하나의 EncoderBlock을 작성한 코드. 위에서 작성한 multi head attention과 mlp 연산을 순서에 맞게 가져와서 사용하면 됨.
class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.calculate_attn = MultiHeadAttention(self.config)  # multi head attention을 위한 함수
        self.feedforward = Mlp(self.config)  # feed forwatd를 위한 함수
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        # multi head attention을 하고 layer norm
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        # feed forwatd를 하고 layer norm

    def forward(self, inputs, attn_mask):
        attn_output, attn_prob = self.calculate_attn(inputs, inputs, inputs, attn_mask)
        attn_output = self.layer_norm1(inputs + attn_output)
        # layer norm 함수 안에는 residual connection 연산을 함. (입력값과 출력값을 더함.)

        forward_output = self.feedforward(attn_output)
        final_output = self.layer_norm2(forward_output + attn_output)

        return final_output, attn_prob


# 하나의 encoder를 정의.
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_embed = nn.Embedding(self.config.n_enc_vocab, self.config.d_hidn)  # 입력문장의 임베딩 벡터를 가져옴.
        pos_val = torch.FloatTensor(calcul_location(self.config.n_enc_seq + 1, self.config.d_hidn))  # 위치 임베딩을 계산.
        self.pos_embed = nn.Embedding.from_pretrained(pos_val, freeze=True)  # 위치 임베딩 값이 고정적이게 함.

        # encoderblock을 한 layer로 가짐.
        self.layers = nn.ModuleList([EncoderBlock(self.config) for _ in range(self.config.n_layer)])

    def forward(self, inputs):
        position = (torch.arange(inputs.size(1), device=inputs.size(), dtype=inputs.dtype).
                    expand(inputs.size(0), inputs.size(1)).contiguous() + 1)
        pos_mask = inputs.eq(self.config.i_pad)  # inputs에 값이 0인 원소가 있으면 pos_mask는 그 자리에 true을 넣고 아니면 false을 넣음.
        position.masked_fill(pos_mask, 0)  # pos_mask가 true인 위치에 0으로 채움.

        output = self.input_embed(inputs) + self.pos_embed(position)  # 워드 임베딩과 위치 임베딩을 더해줌.(따지고보면 입력값)

        attn_probs = []
        attn_mask = padding_mask(inputs, inputs, self.config.i_pad)

        for layer in self.layers:  # encoder의 block에 순서대로 넣음.
            output, attn_prob = layer(output, attn_mask)
            attn_probs.append(attn_prob)

        return output, attn_probs
