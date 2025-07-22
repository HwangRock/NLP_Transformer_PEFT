# Transformer and PEFT

Pytorch로 Transformer를 코드 레벨에서 구현하고,   
대표적인 PEFT인 Adapter, LoRA, P-tuning, Prompt tuning, Prefix tuning 또한 코드레벨에서 구현해서 성능을 몸소 실험하는 레포지토리입니다.

---

## 실험 환경

- GPU : NVIDIA RTX3060
- Language : Python 3.10.12
- Framework : Pytorch
- Model : Pretrained BERT vs. PEFT BERT
- Experiment task : Sentence-level Classification
- Dataset : GLUE Benchmark - SST-2
- Evaluation Metrics: Accuracy, F1-score, GLUE Average

---

## 구현 Task

- Transformer : 2024-08-09 ~ 2024-08-16 구현 완료
- Adapter : 2025-07-14 ~ 2025-07-22 구현 완료
- LoRA
- P-tuning

---

## 이론 정리

논문에 생략된 수학적인 내용도 정확하게 짚고 싶어서 선형대수학 및 확률론의 내용을 세부 설명없이 사용했습니다.  
아래의 링크에서 이론을 볼 수 있습니다.  
</br>
[Transformer](./Transformer/README.md)  
[Adapter](./Adapter/README.md)

