# ParakeetHD: Huntington Disease ASR Model Suite

Official repository for the paper **"Towards Robust Automatic Speech Recognition for Huntington Disease."**

## Summary

ParakeetHD is a model suite for automatic speech recognition (ASR) on speech affected by Huntington disease (HD). In this work, we:

- compare multiple ASR model families on HD speech under a unified evaluation pipeline,
- adapt **Parakeet-TDT** to HD speech using parameter-efficient encoder-side adapters,
- evaluate performance with **WER** and detailed **substitution / deletion / insertion** analysis,
- and study whether **prosodic, phonatory, and articulatory biomarkers** can be used as auxiliary supervision during adaptation.

Our results show that **HD-specific adaptation** gives the strongest overall performance, while biomarker-aware supervision helps reveal clinically meaningful changes in error behavior.

## Models

- **Parakeet-HD**
- **Parakeet-HD-Prosody**
- **Parakeet-HD-Phonation**
- **Parakeet-HD-Articulation**
