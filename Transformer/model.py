import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder


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
