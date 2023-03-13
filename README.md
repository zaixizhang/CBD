# CBD
Official Inplementation of CVPR23 paper "Backdoor Defense via Deconfounded Representation Learning"
<div align=center><img src="https://github.com/zaixizhang/CBD/blob/main/backdoor.png" width="700"/></div>
Deep neural networks (DNNs) are recently shown to be vulnerable to backdoor attacks, where attackers embed hidden backdoors in the DNN model by injecting a few poisoned examples into the training dataset. While extensive efforts have been made to detect and remove backdoors from backdoored DNNs, it is still not clear whether a backdoor-free clean model can be directly obtained from poisoned datasets. In this paper, we first construct a causal graph to model the generation process of poisoned data and find that the backdoor attack acts as the confounder, which brings spurious associations between the input images and target labels, making the model predictions less reliable. Inspired by the causal understanding, we propose the Causality-inspired Backdoor Defense (CBD), to learn deconfounded representations for reliable classification. Specifically, a backdoored model is intentionally trained to capture the confounding effects. The other clean model dedicates to capturing the desired causal effects by minimizing the mutual information with the confounding representations from the backdoored model and employing a sample-wise re-weighting scheme. Extensive experiments on multiple benchmark datasets against 6 state-of-the-art attacks verify that our proposed defense method is effective in reducing backdoor threats while maintaining high accuracy in predicting benign samples. Further analysis shows that CBD can also resist potential adaptive attacks.

You can run CBD by
```
python3 train.py
```

The configurations can be adjusted by modifying config.py

## Cite

If you find this repo to be useful, please cite our paper. Thank you!

```
@inproceedings{
anonymous2023backdoor,
title={Backdoor Defense via Deconfounded Representation Learning},
author={Zhang, Zaixi and Liu, Qi and Wang, Zhicai and Lu, Zepu and Hu, Qingyong},
booktitle={Conference on Computer Vision and Pattern Recognition 2023},
year={2023}
}
```
