# Kurdish-Sorani OCR Benchmark Report

Generated from the completed 10,000-image evaluation on 2026-04-29.

Dataset:

```text
E:\TRDG\data\kurdish_tts_10k_multifont
```

Final result folders:

```text
Tesseract: new_data/results_tesseract_cpu_all
EasyOCR:   new_data/results_easyocr_gpu_all
EasyOCR ar+fa+ur single run: new_data/results_easyocr_ar_plus_fa_plus_ur
PaddleOCR: new_data/results_paddleocr_gpu_v5
Combined:  new_data/results_all_models
```

## Combined Results

```csv
engine,config_id,configuration,n_images,cer_mean,cer_std,wer_mean,wer_std,time_mean_s,time_total_s,coverage_pct
Tesseract CPU,ara,Tesseract (ara),10000,37.630768,17.969115,89.733587,28.228576,0.115287,1152.871164,98.350000
Tesseract CPU,fas,Tesseract (fas),10000,28.093224,17.158592,82.174081,29.616177,0.098362,983.620771,98.350000
Tesseract CPU,urd,Tesseract (urd),10000,20.328423,19.548735,58.750281,32.825237,0.123818,1238.184241,98.350000
Tesseract CPU,ara_plus_fas,Tesseract (ara+fas),10000,31.633810,17.229099,87.410131,29.012804,0.161344,1613.444108,98.350000
Tesseract CPU,fas_plus_urd,Tesseract (fas+urd),10000,26.286338,17.553793,77.915580,30.338639,0.168847,1688.473672,98.350000
Tesseract CPU,ara_plus_urd,Tesseract (ara+urd),10000,29.078743,18.374883,76.651205,30.553511,0.185826,1858.259777,98.350000
Tesseract CPU,fas_no_dawg,"Tesseract (fas, no DAWG)",10000,28.093224,17.158592,82.174081,29.616177,0.098736,987.357058,98.350000
Tesseract CPU,ara_no_dawg,"Tesseract (ara, no DAWG)",10000,37.630768,17.969115,89.733587,28.228576,0.116047,1160.469598,98.350000
EasyOCR GPU,ar,EasyOCR (ar),10000,29.915420,5.371876,91.361989,12.389161,0.113942,1139.421462,100.000000
EasyOCR GPU,fa,EasyOCR (fa),10000,24.814841,5.089771,85.207192,13.563539,0.127117,1271.170541,100.000000
EasyOCR GPU,ur,EasyOCR (ur),10000,20.549540,6.042083,75.063209,16.182561,0.093991,939.908563,100.000000
EasyOCR GPU,ar_plus_fa,EasyOCR (ar+fa),10000,24.814841,5.089771,85.207192,13.563539,0.098268,982.676335,100.000000
EasyOCR GPU,fa_plus_ur,EasyOCR (fa+ur),10000,21.839141,5.592538,78.248954,14.676499,0.089587,895.867483,100.000000
EasyOCR GPU,ar_plus_ur,EasyOCR (ar+ur),10000,21.839141,5.592538,78.248954,14.676499,0.090734,907.338679,100.000000
EasyOCR GPU,ar_plus_fa_plus_ur,EasyOCR (ar+fa+ur),10000,21.839141,5.592538,78.248954,14.676499,0.090496,904.959144,100.000000
PaddleOCR GPU PP-OCRv5,ar,PaddleOCR v5 (ar),10000,25.507608,7.736786,81.803169,13.538772,0.064304,643.037686,99.980000
PaddleOCR GPU PP-OCRv5,fa,PaddleOCR v5 (fa),10000,25.507608,7.736786,81.803169,13.538772,0.065149,651.492008,99.980000
PaddleOCR GPU PP-OCRv5,ur,PaddleOCR v5 (ur),10000,25.507608,7.736786,81.803169,13.538772,0.064561,645.607580,99.980000
```

## Paper-Ready Table

```text
Configuration               Engine                    CER, %    WER, %    T, s    Coverage, %
Tesseract (ara)             Tesseract CPU              37.63     89.73    0.12           98.3
Tesseract (fas)             Tesseract CPU              28.09     82.17    0.10           98.3
Tesseract (urd)             Tesseract CPU              20.33     58.75    0.12           98.3
Tesseract (ara+fas)         Tesseract CPU              31.63     87.41    0.16           98.3
Tesseract (fas+urd)         Tesseract CPU              26.29     77.92    0.17           98.3
Tesseract (ara+urd)         Tesseract CPU              29.08     76.65    0.19           98.3
Tesseract (fas, no DAWG)    Tesseract CPU              28.09     82.17    0.10           98.3
Tesseract (ara, no DAWG)    Tesseract CPU              37.63     89.73    0.12           98.3
EasyOCR (ar)                EasyOCR GPU                29.92     91.36    0.11          100.0
EasyOCR (fa)                EasyOCR GPU                24.81     85.21    0.13          100.0
EasyOCR (ur)                EasyOCR GPU                20.55     75.06    0.09          100.0
EasyOCR (ar+fa)             EasyOCR GPU                24.81     85.21    0.10          100.0
EasyOCR (fa+ur)             EasyOCR GPU                21.84     78.25    0.09          100.0
EasyOCR (ar+ur)             EasyOCR GPU                21.84     78.25    0.09          100.0
EasyOCR (ar+fa+ur)          EasyOCR GPU                21.84     78.25    0.09          100.0
PaddleOCR v5 (ar)           PaddleOCR GPU PP-OCRv5     25.51     81.80    0.06          100.0
PaddleOCR v5 (fa)           PaddleOCR GPU PP-OCRv5     25.51     81.80    0.07          100.0
PaddleOCR v5 (ur)           PaddleOCR GPU PP-OCRv5     25.51     81.80    0.06          100.0
```

Best completed results:

```text
Best CER:     Tesseract (urd), 20.33%
Best WER:     Tesseract (urd), 58.75%
Fastest mean: PaddleOCR v5 (ar), 0.06 s/image
```

## Statistical Comparison

Tesseract paired CER comparisons:

```text
fas vs fas_no_dawg:
  t=0.000000, df=9999, p=1.000000, cohen_d=nan, mean_diff_pp=0.000000

ara vs fas:
  t=195.395913, df=9999, p=0.000e+00, cohen_d=1.953959, mean_diff_pp=9.537544
```

Interpretation:

```text
The Farsi/Persian Tesseract pack significantly outperformed Arabic.
The DAWG-disabled Farsi run was numerically identical to the normal Farsi run, so this experiment does not show a measurable DAWG effect.
```

## Model And Environment Details

Tesseract:

```text
Engine: tesseract v5.3.0.20221214
Device: CPU only
Evaluated language packs:
  ara
  fas
  urd
  ara+fas
  fas+urd
  ara+urd
  fas, no DAWG
  ara, no DAWG
Installed relevant traineddata files:
  ara.traineddata
  fas.traineddata
  urd.traineddata
```

EasyOCR:

```text
EasyOCR version: 1.7.2
PyTorch version: 2.11.0+cu128
CUDA available: True
CUDA version: 12.8
GPU: NVIDIA GeForce RTX 3090
Decoder: greedy
Paragraph grouping: True
Model files used:
  craft_mlt_25k.pth
  arabic.pth
Language lists:
  ar
  fa
  ur
  ar+fa
  fa+ur
  ar+ur
  ar+fa+ur
```

PaddleOCR:

```text
PaddleOCR version: 3.5.0
PaddlePaddle version: 3.3.1
OCR version: PP-OCRv5
Device: gpu:0
Compiled with CUDA: True
GPU: NVIDIA GeForce RTX 3090
Model files used:
  PP-OCRv5_server_det
  arabic_PP-OCRv5_mobile_rec
Language settings:
  ar
  fa
  ur
```

## Hypotheses

H1. `fas` < `ara` in CER:

```text
Supported. Tesseract fas improves over ara: 28.09% vs 37.63% CER.
Also supported in EasyOCR: fa improves over ar: 24.81% vs 29.92% CER.
```

H2. `fas_no_dawg` < `fas` in CER:

```text
Not supported. Tesseract fas_no_dawg and fas are identical at 28.09% CER.
```

H3. Multi-language mode is not strictly better than the closest single language:

```text
Supported. Tesseract ara+fas is worse than fas alone.
EasyOCR ar+fa, fa+ur, ar+ur, and ar+fa+ur do not improve over EasyOCR ur.
```

H4. No configuration achieves CER < 30%:

```text
Refuted. Several configurations are below 30% CER.
Best overall is Tesseract urd at 20.33% CER.
Best neural OCR row is EasyOCR ur at 20.55% CER.
```

## Recommended Paper Wording

```text
On the 10,000-image Kurdish-Sorani synthetic benchmark, the strongest overall result was obtained by Tesseract with the Urdu language pack (CER 20.33%, WER 58.75%). EasyOCR with the Urdu language list was the strongest neural OCR configuration (CER 20.55%, WER 75.06%). PaddleOCR PP-OCRv5 produced identical aggregate scores for ar, fa, and ur (CER 25.51%, WER 81.80%) and was the fastest engine, averaging approximately 0.06 seconds per image on GPU. Multi-language configurations did not improve over the best single-language setting. Disabling Tesseract's Farsi DAWG dictionary did not change the measured score in this run.
```
