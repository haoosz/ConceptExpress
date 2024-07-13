# ConceptExpress
![License](https://img.shields.io/github/license/haoosz/ConceptExpress?color=lightgray)
[![arXiv](https://img.shields.io/badge/arXiv-2407.07077%20-b31b1b)](http://arxiv.org/abs/2407.07077)

This is the official PyTorch codes for the paper:  

[**ConceptExpress: Harnessing Diffusion Models for Single-image Unsupervised Concept Extraction**](http://arxiv.org/abs/2407.07077)  
[Shaozhe Hao](https://haoosz.github.io/),
[Kai Han](https://www.kaihan.org/), 
[Zhengyao Lv](https://scholar.google.com/citations?user=FkkaUgwAAAAJ),
[Shihao Zhao](https://shihaozhaozsh.github.io/),
[Kwan-Yee K. Wong](http://i.cs.hku.hk/~kykwong/)  
ECCV 2024

<p align="left">
    <img src='src/teaser.gif' width="90%">
</p>

*We present **Unsupervised Concept Extraction (UCE)** that focuses on the *unsupervised* problem of extracting *multiple* concepts from a *single* image.*

[**Project Page**](https://haoosz.github.io/ConceptExpress/)

### The dataset of input images used in our paper is now available at [this link](https://drive.google.com/drive/folders/1TJLH15CkIcnPPJJkV8nk3bgnybdRgBrx?usp=sharing). All images in this dataset are sourced from [Unsplash](https://unsplash.com/) under a [license](https://unsplash.com/license) that allows free download and use!

## Set-up
Create a conda environment `uce` using
```
conda env create -f environment.yml
conda activate uce
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Training
Create a new folder that contains an `img.jpg`. For example, download [our dataset](https://drive.google.com/drive/folders/1TJLH15CkIcnPPJJkV8nk3bgnybdRgBrx?usp=sharing) and put it under the root path. You can change `--instance_data_dir` in bash file `scripts/train.sh` to `uce_images/XX` or any other image path you like. You can specify `--output_dir` to save the checkpoints. 

When the above is ready, run the following to start training:
```
bash scripts/train.sh
```
The learned token embeddings of all concepts are saved to `.bin` files under your `--output_dir`.

## Inference
Once trained, the *i*-th concept is represented as `<asset$i>` in the tokenizer. We can then freely generate images using any concept token `<asset$i>` (replace `$i` with a valid concept index):
```
python infer.py \
  --embed_path $CKPT_BIN_FILE \
  --prompt "a photo of <asset$i> in the snow" \
  --save_path $SAVE_FOLDER \
  --seed 0
```
Please specify `$CKPT_BIN_FILE` which is the `.bin` file path of your learned token embeddings, and `$SAVE_FOLDER` to save the generated images. You can also find inference examples in `scripts/infer.sh`.

## Citation
If you use this code in your research, please consider citing our paper:
```bibtex
@InProceedings{hao2024conceptexpress,
    title={Concept{E}xpress: Harnessing Diffusion Models for Single-image Unsupervised Concept Extraction}, 
    author={Shaozhe Hao and Kai Han and Zhengyao Lv and Shihao Zhao and Kwan-Yee K. Wong},
    booktitle={ECCV},
    year={2024},
}
```

## Acknowledgements
This code repository is based on the great work of [Break-A-Scene](https://github.com/google/break-a-scene). Thanks!