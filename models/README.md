# OCR Model Checkpoints

This folder is intentionally empty in git. Put external OCR checkpoints here when you want those engines enabled.

Recommended local layout:

```text
models/
  kraken/
    kraken_arabic_historical_v1.mlmodel
  calamari/
    calamari_arabic_v1.ckpt
```

`config.yaml.example` does not require these files. It lets Kraken run line detection only and keeps Calamari disabled.

`config.full.example` expects both checkpoints. Copy it to `config.yaml` only after placing real model files at the configured paths, or edit the paths to your downloaded checkpoints.
