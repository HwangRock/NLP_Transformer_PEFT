# Transformer and PEFT

Pytorch로 Transformer를 코드 레벨에서 구현하고,   
대표적인 PEFT인 Adapter, LoRA, P-tuning, Prompt tuning, Prefix tuning 또한 코드레벨에서 구현해서 성능을 몸소 실험하는 레포지토리입니다.

---

## 실험 환경

- GPU : NVIDIA RTX3060
- Language : Python 3.10.12
- Framework : Pytorch
- Model : vanila transformer
- Experiment task : SST-2, MNLI, QNLI
- Model matrix: Accuracy, F1, GLUE avg

---

## 구현 Task
- Transformer : 2024-08-09 ~ 2024-08-16 구현 완료
- Adapter
- LoRA
- P-tuning
- Prompt tuning
- Prefix tuning

---

## 이론 정리
논문에 생략된 수학적인 내용도 정확하게 짚고 싶어서 선형대수학 및 확률론의 내용을 세부 설명없이 사용했습니다.  

[Transformer](###Transformer)

### Transformer

#### 주요 개념
- Self-Attention
- Multi-Head Attention
- Positional Encoding
- Masking
- Encoder-Decoder Architecture

#### 정리

##### self-Attention
###### Why use?
트랜스포머 논문 제목이 "Attention is all yod need."라는 다소 폭력적인 제목인만큼 트랜스포머에서는 Attention 기법을 매우 강조합니다.  
그 이유는 이전까지 해결되지 않았던 RNN의 long-term dependency problem을 해결했기 때문입니다.  
</br>
이 문제는 RNN의 계산 구조에서 비롯됩니다. 아래와 같은 recurrence relation을 보면,</br>
$$
h_t = \phi(W_h h_{t-1} + W_x x_t + b)
$$</br>
여기서 현재 hidden state $h_t$는 오직 직전 hidden state $h_{t-1}$에만 의존합니다.  
이러한 재귀적 구조는 시간이 지남에 따라 과거 정보가 희석되고, 긴 시퀀스에서는 정보가 소실되는 현상으로 이어집니다.  
</br>
Self-attention은 이와 달리, 코사인 유사도를 기반으로 한 내적 계산으로 모든 토큰 쌍 간의 유사도를 직접 계산하고, 각 토큰이 전체 시퀀스를 동시에 바라볼 수 있도록 설계되어 있습니다.  
이 구조 덕분에 long-term dependency 문제 없이, 문맥 전체를 한 번에 고려하는 계산이 가능해졌습니다.
<br>

###### How operate?