#### PLGMamba | Journal of Remote Sensing 2025 | HSI-SR
---
## PLGMamba: A New Progressive Local-Global State-Space Model for Hyperspectral Image Super-Resolution

***Chengle Zhou, Zhi He, Li Wang, Jinchang Ren, and Antonio Plaza***

*Journal of Remote Sensing, Accepted, 2025*

---

![framework](https://github.com/chengle-zhou/MY-IMAGE/blob/432c7cc4122c2bd367b5fde76bfcba7ab9229224/PLGMamba/PLGMamba.png)

Figure 1: Overall architecture of the proposed PLGMamba for the HSI-SR task.



## Abstract

Convolutional neural networks (CNNs) and Transformer architectures are among the most popular techniques for hyperspectral image super-resolution (HSI-SR). However, both architectures suffer from inherent deficiencies: CNNs are constrained to a limited receptive field, which hinders their ability to capture a wider range of spatial contexts, while Transformers are computationally intensive, making them expensive to train and deploy in large-scale scenes. Furthermore, existing methods tend to employ end-to-end spectral and spatial modeling paradigms for the HSI-SR task while neglecting the natural spectral patterns of high-dimensional HSIs. To overcome the above problems, this paper proposes a new progressive local-global state space model (SSM) for HSI-SR, named PLGMamba, which fully takes into account feature extraction and model design from a local to a global perspective. Initially, a progressive learning paradigm to extract spectral sequences (from local to global) is applied to the low-resolution HSI (LR-HSI) to ensure spatial and spectral fidelity. Then, a residual attention Mamba (RatMamba) is developed to capture the local-global spectral-spatial features of the reconstructed HSI, which integrates spectral-spatial local perception and global modeling into a unified residual learning framework. Finally, a residual Mamba (ResMamba) is designed to fuse progressive local spectral-spatial features and obtain the final high-resolution HSI (HR-HSI). Experiments on three benchmark HSIs and one GF-5 HSI demonstrated that the proposed PLGMamba outperforms classical and state-of-the-art HSI-SR approaches including the mainstream CNN, Transformer, and Mamba.


***We are working on it and will release it soon.***
