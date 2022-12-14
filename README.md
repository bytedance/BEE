## BEE (ByteDance End-to-End) reference software for IEEE 1857.11 Standard for Neural Network-Based Image Coding

The software is the reference software for IEEE 1857.11 standard for neural network-based image coding, 
including encoder, decoder and training functionalities. 

### Environment

The software requires PyTorch version >= 1.9.0 and the following packages

```text
scipy
scikit-image
torchvision
numpy
matplotlib
lmdb
opencv-python-headless
openpyxl
einops
pyrtools
pytorch-msssim
IQA-pytorch
psnr_hvsm
ptflops==0.6.5
```

### Encoding and Decoding

-----------

Please download the pretrained models **[updated on Aug 30, 2022]** (under folder ``pretrained_models``) and the metric weights (under folder ``Metric``) here [https://pan.baidu.com/s/1vW9emHqqHrDZJ1abS4UCxA](https://pan.baidu.com/s/1vW9emHqqHrDZJ1abS4UCxA) (Extraction code: dwc0).

- Copy Metric/metric_tool/weights to Metric/metric_tool


**A single image-->**

```bash
# Encode using configuration file
python3 Encoder/CoreEncApp.py -i IEEE1857TestSet/01.png -o str.bin --ckptdir pretrained_models/objective/enc_model --qp 46 --cfg Encoder/IEEE_AllRecipes_Objective.json --oldversion

# Encode using a specific checkpoint file
python3 Encoder/CoreEncApp.py -i IEEE1857TestSet/01.png -o str.bin --ckpt pretrained_models/objective/enc_model/quant_model.ckpt-02 --qp 46 --oldversion

# Decode
python3 Decoder/DecApp.py -i bitstreams/str.bin -o rec.png --ckptdir pretrained_models/objective/dec_model --oldversion
```

**All images under a folder-->**

```bash
# Encode using configuration file
python3 Encoder/CoreEncApp.py --inputPath ./IEEE1857TestSet --outputPath ./bin --ckptdir pretrained_models/objective/enc_model --qp 46 --cfg Encoder/IEEE_AllRecipes_Objective.json --oldversion

# Encode using a specific checkpoint file
python3 Encoder/CoreEncApp.py --inputPath ./IEEE1857TestSet --outputPath ./bin --ckpt pretrained_models/objective/enc_model/quant_model.ckpt-02 --qp 46 --oldversion

# Decode
python3 Decoder/DecApp.py --binpath ./bin --recpath ./rec --ckptdir pretrained_models/objective/dec_model --oldversion
```

### Training

-----------

Stage1 training example:
```bash
python3 Train/run.py Train/trainYUV.sh -c Train/cfg/TrainConfigStage1 --quality 2 --checkpoint Stage1/Q2 
```

> Note: --quality could be one from [2, 4, 6, 8, 10]

Stage2 training example:
```bash
python3 Train/run.py Train/trainYUV.sh -c Train/cfg/TrainConfigStage2 --quality 2 --InitModel Stage1/Q2/best.pth --checkpoint Stage2/Q2 --learning_rate 1e-5 
```

> Note: --InitModel is the pretrained model from Stage1, choose --quality from [2, 4, 6, 8, 10]

Stage3 training example:
```bash
python3 Train/run.py Train/trainYUV.sh -c Train/cfg/TrainConfigStage3 --quality 1 --InitModel Stage2/Q2/best.pth --checkpoint Stage3/Q1 --learning_rate 1e-5 
```

> Note: choose --quality from [1, 3, 5, 7, 9, 11, 12, 13, 14, 15, 16]
> 
> --InitModel is the pretrained model from Stage2, use the following pretrained models from stage-2
> 
> (--quality stage3, InitModel stage2) 
> 
> (1, Stage2/Q2/best.pth), (3, Stage2/Q4/best.pth), (5, Stage2/Q6/best.pth), (7, Stage2/Q8/best.pth), (9, Stage2/Q10/best.pth)
> 
> All use Stage2/Q10/best.pth for --quality in 11 - 16.

### License

-----------

BEE is licensed under the Apache License, Version 2.0

### Contacts

-----------

- Semih Esenlik, semih.esenlik@bytedance.com
- Yaojun Wu, wuyaojun@bytedance.com
- Zhaobin Zhang, zhaobin.zhang@bytedance.com

