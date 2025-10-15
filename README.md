# Audio-llm project

ASR performance (CER %)
||Ksponspeech <br> eval-clean|Ksponspeech <br> eval-other|
|:----:|:---:|:---:|
|w/ text normalization|7.52|7.85|
|w/o text normalization|7.20|7.30|

Speech-to-text translation performance (BLEU)
|kosp2e ko→en|
|:------:|
|24.35|

* 학습 및 테스트 데이터셋에 Whisper normalizer 텍스트 정규화를 적용했습니다.
