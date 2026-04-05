<div align="center">

# 🦜 ParakeetHD: Huntington Disease ASR Model Suite

[![arXiv](https://img.shields.io/badge/arXiv-2603.11168-b31b1b.svg)](https://arxiv.org/abs/2603.11168)
[![Hugging Face Collection](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-FFD21E)](https://huggingface.co/collections/charleslwang/parakeethd)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

*Official repository for the paper **"Huntington Disease Automatic Speech Recognition with Biomarker Supervision."***

</div>

---

## 📌 Quick Links
- [Summary](#-summary)
- [Models](#-models)
- [Citation](#-citation)

---

## 📖 Summary

**ParakeetHD** is a model suite designed for automatic speech recognition (ASR) on speech affected by Huntington disease (HD). In this work, we:

*   🔍 **Compare** multiple ASR model families on HD speech under a unified evaluation pipeline.
*   🛠️ **Adapt** **Parakeet-TDT** to HD speech using parameter-efficient encoder-side adapters.
*   📊 **Evaluate** performance with **WER** and detailed **substitution / deletion / insertion** analysis.
*   🩺 **Study** whether **prosodic, phonatory, and articulatory biomarkers** can be used as auxiliary supervision during adaptation.

> **Key Finding:** Our results show that **HD-specific adaptation** gives the strongest overall performance, while biomarker-aware supervision helps reveal clinically meaningful changes in error behavior.

---

## 🤖 Models

All models in the suite are hosted in our [Hugging Face Collection](https://huggingface.co/collections/charleslwang/parakeethd). 

| Model | Focus / Supervision | Link |
| :--- | :--- | :--- |
| **Parakeet-HD** | Baseline HD Adaptation | [View Model](https://huggingface.co/collections/charleslwang/parakeethd) |
| **Parakeet-HD-Prosody** | Prosodic Biomarkers | [View Model](https://huggingface.co/charleslwang/parakeet-tdt-0.6b-HD-prosody) |
| **Parakeet-HD-Phonation** | Phonatory Biomarkers | [View Model](https://huggingface.co/charleslwang/parakeet-tdt-0.6b-HD-phonation) |
| **Parakeet-HD-Articulation**| Articulatory Biomarkers | [View Model](https://huggingface.co/charleslwang/parakeet-tdt-0.6b-HD-articulation) |

---

## 📝 Citation

If you use this repository or the released models, please cite our paper:

```bibtex
@article{wang2026huntington,
  title={Huntington Disease Automatic Speech Recognition with Biomarker Supervision},
  author={Wang, Charles L. and Chen, Cady and Gong, Ziwei and Hirschberg, Julia},
  journal={arXiv preprint arXiv:2603.11168},
  year={2026},
  url={[https://arxiv.org/abs/2603.11168](https://arxiv.org/abs/2603.11168)}
}
