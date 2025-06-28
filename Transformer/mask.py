import torch


# padding된 토큰을 무시하기 위한 함수.
def padding_mask(input_q, input_k, pad_num):
    batch_size, len_q = input_q.size()
    batch_size, len_k = input_k.size()

    pad = input_k.detach().eq(pad_num).unsqueeze(1).expand(batch_size, len_q, len_k)

    return pad


# decoder에서 보지 않는 단어는 mask하기 위한 함수.
def decoder_mask(seq):
    mask = torch.ones_like(seq).unsqueeze(-1).expand(seq.size(0), seq.size(1), seq.size(1))
    mask = mask.triu(diagonal=1)  # 행렬의 주대각선 위는 1로 채우고, 나머지는 0으로 채워서 다음 토큰을 마스크하게 함.
    return mask
