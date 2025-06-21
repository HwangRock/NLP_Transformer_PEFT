import numpy as np


# sinusodial에 기반해서 위치 임베딩을 계산하는 함수
def calcul_location(length, dimension):
    def angle(position, index):
        return position / np.power(10000, 2 * index / dimension)  # 삼각함수 내부에 들어갈 각을 계산해줌.

    def calcul_list(position):
        return [angle(position, i) for i in range(dimension)]  # 원소의 수를 은닉층의 차원만큼 형성하여 배열을 형성. 값은 삼각함수에 들어갈 값

    pos_embed = np.array([calcul_list(i) for i in range(length)])  # calcul_list를 이용해서 n_seq * d_hidn 크기의 2차원 배열을 제작.
    pos_embed[:, 0::2] = np.sin(pos_embed[:, 0::2])  # 짝수번째는 sin을 이용하여 값을 형성
    pos_embed[:, 1::2] = np.cos(pos_embed[:, 1::2])  # 홀수번째는 cos을 이용하여 값을 형성

    return pos_embed
