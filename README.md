# PR-NN_Detector_PRchannel
PR-NN: RNN-based Detection for Coded Partial-Response Channels

This repository contains the source code presented in the following paper: PR-NN: RNN-based Detection for Coded Partial-Response Channels (https://ieeexplore.ieee.org/document/9279252 or https://arxiv.org/pdf/2007.15695.pdf) by Simeng Zheng, Yi Liu, and Paul Siegel.

## Usage
Install all required library (PyTorch):

```
conda env create -f environment.yml --name pr-nn
```

## Individual Training

## Joint Training

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
