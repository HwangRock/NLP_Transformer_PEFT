# Transformer and PEFT

Pytorch로 Transformer를 코드 레벨에서 구현하고,   
대표적인 PEFT인 Adapter, LoRA, P-tuning, Prompt tuning, Prefix tuning 또한 코드레벨에서 구현해서 성능을 몸소 실험하는 레포지토리입니다.

---

## 실험 환경

- GPU : NVIDIA RTX3060
- Language : Python 3.10.12
- Framework : Pytorch
- model : vanila transformer
- Expriment task : SST-2, MNLI, QNLI
- model matrix: Accuracy, F1, GLUE avg

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
[Transformer](###Transformer)

### Transformer

##### 주요 개념
- Self-Attention
- Multi-Head Attention
- Positional Encoding
- Masking
- Encoder-Decoder Architecture