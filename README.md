# PR-NN_Detector_PRchannel
PR-NN: RNN-based Detection for Coded Partial-Response Channels

This repository contains the source code presented in the following paper: PR-NN: RNN-based Detection for Coded Partial-Response Channels (https://ieeexplore.ieee.org/document/9279252 or https://arxiv.org/pdf/2007.15695.pdf) by Simeng Zheng, Yi Liu, and Paul Siegel.

## Usage
Install all required library (PyTorch):

```
conda env create -f environment.yml --name pr-nn
```

## Individual Training

We individually train PR-NN under AWGN or Additive colored noise (ACN) and evaluate PR-NN under this distortion. The colored noise is generated from the PR equalizer in magnetic recording model. We compare the results in PR-NN with Viterbi, BCJR, and NPML. Additionally, we design the sliding block decoder (SBD) to recorver the data in the recording model. 

The evaluation length suggested (SNR from 8.5 dB to 10.5 dB) is 1E06. We guarantee that the observed errors are at least 100 for each noise and SNR. If you want to enhance the length of evaluation sequence, you should also increase the training epoch and enlarge the training dataset.

## Joint Training

Because the channel parameters in a magnetic hard disk storage system can vary, we design the joint training to see the robustness of PR-NN. We feed PR-NN with two kinds of distortions (two ACNs with different parameters) and evaluate PR-NN under these distortions. 

In this part, we mainly consider two cases: 1) AWGN+ACN; 2) ACN+ACN (with different channel parameters). You can change the ratio of AWGN and ACN by changing the batch size for each noise.

```
python main_white_color.py -batch_size_snr_train_white 30 -batch_size_snr_train_color 30
```

```
python main_color.py -batch_size_snr_train_1 30 -batch_size_snr_train_2 30
```

## Training in 'Realistic' Systems

## Classical Detectors

There are three main classical signal detection methods for the magnetic recording system: Viterbi detector, BCJR detector, NPML (Noise-predictive maximum-likelihood) detector. We provide the implemention codes in Python.

1.Viterbi - Error bounds for convolutional codes and an asymptotically optimum decoding algorithm, by Andrew Viterbi.

```
python viterbi.py -info_len 1000000 -snr_start 8.5 -snr_stop 10.5
```

2.BCJR (max-log-map and sliding-window implementation) - An intuitive justification and a simplified implementation of the MAP decoder for convolutional codes, by Andrew Viterbi 

```
python bcjr.py -info_len 1000000 -snr_start 8.5 -snr_stop 10.5
```

3.NPML - Noise-predictive maximum likelihood (NPML) detection, by Jonathan D. Coker

```
python npml.py -info_len 1000000 -snr_start 8.5 -snr_stop 10.5
```

## Emails
Simeng Zheng - sizheng@eng.ucsd.edu

Paul Siegel - psiegel@eng.ucsd.edu
