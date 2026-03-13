# ParakeetHD: Huntington Disease ASR Model Suite

Official repository for the paper **"Huntington Disease Automatic Speech Recognition with Biomarker
Supervision."**

## Summary

ParakeetHD is a model suite for automatic speech recognition (ASR) on speech affected by Huntington disease (HD). In this work, we:

- compare multiple ASR model families on HD speech under a unified evaluation pipeline,
- adapt **Parakeet-TDT** to HD speech using parameter-efficient encoder-side adapters,
- evaluate performance with **WER** and detailed **substitution / deletion / insertion** analysis,
- and study whether **prosodic, phonatory, and articulatory biomarkers** can be used as auxiliary supervision during adaptation.

Our results show that **HD-specific adaptation** gives the strongest overall performance, while biomarker-aware supervision helps reveal clinically meaningful changes in error behavior.

## Models

Models can be found here: https://huggingface.co/collections/charleslwang/parakeethd

- **Parakeet-HD**
- **Parakeet-HD-Prosody**
- **Parakeet-HD-Phonation**
- **Parakeet-HD-Articulation**

## Citation

If you use this repository or the released models, please cite:

```bibtex
@article{wang2026huntington,
  title={Huntington Disease Automatic Speech Recognition with Biomarker Supervision},
  author={Wang, Charles L. and Chen, Cady and Gong, Ziwei and Hirschberg, Julia},
  journal={arXiv preprint arXiv:2603.11168},
  year={2026},
  url={https://arxiv.org/abs/2603.11168}
}
