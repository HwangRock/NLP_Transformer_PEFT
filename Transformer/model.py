import torch
import torch.nn as nn
import torch.nn.functional as f
from model.embedding import calcul_location
from mask import padding_mask, decoder_mask
from model.attention import MultiHeadAttention
from model.layers import Mlp


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


# Decoder에서 사용할 하나의 DecoderBlock을 작성한 코드. 마찬가지로 순서에 맞게 가져온다.
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.masked_attn = MultiHeadAttention(self.config)  # multi head attention을 위한 함수
        self.dec_enc_attn = MultiHeadAttention(self.config)  # multi head attention을 위한 함수
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        # masked multi head attention하고 정규화를 위함.
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        # encoder-decoder multihead attention하고 정규화를 위함.
        self.layer_norm3 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        # Feed forward neural net을 하고 정규화를 위함.
        self.feedforward = Mlp(self.config)  # feed forward를 위한 함수

    def forward(self, dec_inputs, enc_outputs, attn_mask, dec_enc_attn_mask):
        attn_output, attn_prob = self.masked_attn(dec_inputs, dec_inputs, dec_inputs, attn_mask)
        attn_output = self.layer_norm1(dec_inputs + attn_output)
        dec_enc_output, dec_enc_attn_prob = self.dec_enc_attn(attn_output, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_enc_output = self.layer_norm2(dec_enc_output + attn_output)
        output = self.feedforward(dec_enc_output)
        output = self.layer_norm3(output + dec_enc_output)

        return output, attn_prob, dec_enc_attn_prob


# 하나의 decoder를 정의.
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.input_embed = nn.Embedding(self.config.n_enc_vocab, self.config.d_hidn)  # 입력문장의 임베딩 벡터를 가져옴.
        pos_val = torch.FloatTensor(calcul_location(self.config.n_enc_seq + 1, self.config.d_hidn))  # 위치 임베딩을 계산.
        self.pos_embed = nn.Embedding.from_pretrained(pos_val, freeze=True)  # 위치 임베딩 값이 고정적이게 함.

        # decoderblock을 한 layer로 가짐.
        self.layers = nn.ModuleList([DecoderBlock(self.config) for _ in range(self.config.n_layer)])

    def forward(self, inputs, enc_outputs):
        position = (torch.arange(inputs.size(1), device=inputs.size(), dtype=inputs.dtype).
                    expand(inputs.size(0), inputs.size(1)).contiguous() + 1)
        pos_mask = inputs.eq(self.config.i_pad)  # inputs에 값이 0인 원소가 있으면 pos_mask는 그 자리에 true을 넣고 아니면 false을 넣음.
        position.masked_fill(pos_mask, 0)  # pos_mask가 true인 위치에 0으로 채움.

        output = self.input_embed(inputs) + self.pos_embed(position)  # 워드 임베딩과 위치 임베딩을 더해줌.(따지고보면 입력값)
        dec_token_padding = padding_mask(inputs, inputs, self.config.i_pad)
        dec_next_padding = decoder_mask(inputs)
        dec_mask = torch.gt((dec_token_padding or dec_next_padding), 0)
        dec_enc_mask = padding_mask(inputs, enc_outputs, self.config.i_pad)

        masked_probs = []
        dec_enc_probs = []
        for layer in self.layers:
            output, masked_prob, dec_enc_prob = layer(output, enc_outputs, dec_mask, dec_enc_mask)
            masked_probs.append(masked_prob)
            dec_enc_probs.append(dec_enc_prob)

        return output, masked_probs, dec_enc_probs


# 전체적인 transformer의 틀. encoder를 실행하고 decoder를 실행함.
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs, enc_prob = self.encoder(enc_inputs)

        dec_outputs, dec_prob, dec_enc_prob = self.decoder(dec_inputs, enc_outputs)

        return dec_outputs, enc_prob, dec_prob, dec_enc_prob
