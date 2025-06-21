import torch
import torch.nn as nn


# Q, K, V를 다 구했을때 attention을 구하기 위한 클래스.
class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scale = 1 / self.config.d_head ** 0.5  # 차원의 제곱근으로 나눔. 정규화의 과정.

    def forward(self, q, k, v, attn_mask):
        score_matrix = torch.matmul(q, k.transpose(-1, -2)).mul(self.scale)
        # Q와 K의 내적을 해서 Query에 대한 답을 듣고, 내적한 결과에 정규화를 해줌.

        score_matrix.masked_fill_(attn_mask, -1e9)  # masking이 된 값에 음수를 넣어서 softmax를 한 후 값이 0이 되게 한다.
        attn_prob = nn.Softmax(dim=-1)(score_matrix)  # softmax를 사용해서 확률값으로 변환
        attention_matrix = torch.matmul(attn_prob, v)  # score_maxtrix와 V를 내적해서 attention_matrix를 구해서 어텐션 결과를 구함.

        return attention_matrix, attn_prob


# 입력으로부터 Q, K, V를 다 구하고, self attention으로부터 context를 구해온다.
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 입력 텐서를 각각 Q, K, V로 만들기 위한 선형 레이어.
        self.Q_weight = nn.Linear(self.config.d_hidn, self.config.d_head * self.config.n_head)
        self.K_weight = nn.Linear(self.config.d_hidn, self.config.d_head * self.config.n_head)
        self.V_weight = nn.Linear(self.config.d_hidn, self.config.d_head * self.config.n_head)

        self.scale_dot_product = SelfAttention(self.config)  # 가져온 Q, K, V를 사용해서 self attention을 계산하기 위한 함수를 가져옴.
        self.linear = nn.Linear(self.config.d_head * self.config.n_head, self.config.d_hidn)  # 계산한 Q,K,V의 차원을 줄이기 위함.

    def forward(self, pre_q, pre_k, pre_v, attn_mask):
        batch_size = pre_q.size(0)  # 한번에 처리할 입력 데이터 수

        q = self.Q_weight(pre_q).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        k = self.K_weight(pre_k).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        v = self.V_weight(pre_v).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1, 2)
        # batch_size만큼 헤드를 분리하고, 차원에 따라 인식해서 혼자 처리함. 행렬의 열은 n_head, 행은 d_head임.
        # view까지만하면 메모리 내의 순서는 바뀌지 않아서 계산이 독립적으로 수행이 안됨. transpose 연산을 이용해서 차원 순서를 바꿔서 연산을 독립적으로 수행.

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)
        # attn_mask도 멀티 헤드 해주는 연산. unsqueeuze를 해줘서 1번째에 1 크기의 벡터를 추가하고
        # repeat을 통해서 (batch_size, n_head, 차원, 차원)으로 attn_mask의 차원을 변환.

        context, attn_prob = self.scale_dot_product(q, k, v, attn_mask)  # 계산한 Q, K, V를 셀프 어텐션 연산시킴.
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.d_head * self.config.n_head)
        # attention matrix들을 concatenate하는 부분.

        context = self.linear(context)  # attention head의 원래 크기로 되돌리기 위해 W0를 곱해서 크기를 줄임.

        return context, attn_prob
