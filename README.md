# Fingerprint Liveness Detection

## What's implemented

|  | Filename | Description |
|---|---|---|
| Feature extraction | `extract_features.py` | Convnet-features (CNN) |
| Data augmentation | `augment_data.py` | Horizontal flip, crop 5+5 patches |
| Models | `convnet_svm.py` | SVM |
|  | `convnet_nn.py` | Neural network |
|  | `imagenet_finetune.py` | Inception v3 |

CNN feature extraction requires [CNN-RFW](https://github.com/giovanichiachia/convnet-rfw).

Inception v3 settings: samples_per_epoch=250, nb_epoch=25.

## Results

| Pipeline | ACC | ACE |
|---|---|---|
| BSIF + NN | 85.24 | 14.28 |
| AUG + BSIF + SVM | 84.86 | 14.44 |
| AUG + BSIF + NN | 85.93 | 13.82 |
| CNN-RFW + SVM | 81.16 | 17.95 |
| CNN-RFW + NN | 81.88 | 18.16 |
|  |  |  |
| Inception v3 | 66.60 | 28.93 |
|  |  |  |
| BSIF/CNN-RFW + NN | 83.24 | 17.15 |

Average classification error: **ACE** = (FPR + FNR)/2

Inception v3 with ImageNet weights couldn't perform well for our peculiar images of the fingerprints.

**BSIF/CNN-RFW** means mixed features.

## Links

[LivDet 2015 Fingerprint Database](http://livdet.org/registration.php)

[The group project](https://github.com/Guiliang/FingerprintLivenessDetection-project)

## References

[LivDet 2015 Fingerprint Liveness Detection Competition](https://www.clarkson.edu/sites/default/files/2018-01/LivDet%202015.pdf)

[Review of the LivDet Competition Series: 2009 to 2015](https://arxiv.org/abs/1609.01648)

[Evaluating software-based fingerprint liveness detection using Convolutional Networks and Local Binary Patterns](https://arxiv.org/abs/1508.00537)

D. Maltoni, D. Maio, A. Jain, and S. Prabhakar. *Handbook of Fingerprint Recognition.* Springer Publishing Company, 2009.
