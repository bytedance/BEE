# BEE Codebase

This is a codebase of BEE.

## Documentation

### Examples of Training models

Stage1 training example:
```bash
bash Train/trainYUV.sh -c Train/cfg/TrainConfigStage1 --quality 2 --checkpoint Stage1/Q2 
```

> Note: --quality could be [2, 4, 6, 8, 10]

Stage2 training example:
```bash
bash Train/trainYUV.sh -c Train/cfg/TrainConfigStage2 --quality 2 --InitModel Stage1/Q2/best.pth --checkpoint Stage2/Q2 --learning_rate 1e-5 
```

> Note: --InitModel is the pretrained model from Stage1, choose --quality from [2, 4, 6, 8, 10]

Stage3 training example:
```bash
bash Train/trainYUV.sh -c Train/cfg/TrainConfigStage3 --quality 1 --InitModel Stage2/Q2/best.pth --checkpoint Stage3/Q1 --learning_rate 1e-5 
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
> 

### Example of Test(Encode/Decoder)

Encode/Decode a single file:

```bash
# Encode using configuration file
python3 Encoder/CoreEncApp.py -i org.png -o str.bin --ckptdir checkpoints --target_rate 0.06 --cfg Encoder/AllRecipesFinal_objective.json

# Encode using a specific checkpoint file
python3 Encoder/CoreEncApp.py -i org.png -o str.bin --ckpt checkpoints/model.ckpt-02 --target_rate 0.06 

# Decode
python3 Decoder/DecApp.py -i str.bin -o rec.png --ckptdir checkpoints 
```

> Note: put all 16 models under folder checkpoints and rename them to model.ckpt-{k:02d}, k = [1,2,...16]

Encode/Decode all files under a folder:

```bash
# Encode using configuration file
python3 Encoder/CoreEncApp.py --inputPath ./org --outputPath ./bin --ckptdir checkpoints --target_rate 0.06 --cfg Encoder/AllRecipesFinal_objective.json

# Encode using a specific checkpoint file
python3 Encoder/CoreEncApp.py --inputPath ./org --outputPath ./bin --ckpt checkpoints/model.ckpt-02 --target_rate 0.06 

# Decode
python3 Decoder/DecApp.py --binpath ./bin --recpath ./rec --ckptdir checkpoints 
```

## License

BEE is licensed under the Apache License, Version 2.0