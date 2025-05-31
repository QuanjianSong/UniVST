<div align="center">
<h1>
UniVST: A Unified Framework for Training-free Localized Video Style Transfer [Official Code of PyTorch]
</h1>

<div>
    <a href='https://github.com/QuanjianSong' target='_blank' style='text-decoration:none'>Quanjian Song<sup>1</sup></a>, &ensp;
    <a href='https://scholar.google.com/citations?user=Dp3L1bsAAAAJ&hl=zh-CN&oi=ao' target='_blank' style='text-decoration:none'>Mingbao Lin<sup>2</sup></a>, &ensp;
    <a href='https://scholar.google.com/citations?hl=zh-CN&user=nnF1s7kAAAAJ' target='_blank' style='text-decoration:none'>Wengyi Zhan<sup>1</sup></a>, &ensp;
    <a href='https://scholar.google.com/citations?user=DNuiPHwAAAAJ&hl=zh-CN&oi=ao' target='_blank' style='text-decoration:none'>Shuicheng Yan<sup>2</sup></a>, &ensp;
    <a href='https://mac.xmu.edu.cn/ljcao/' target='_blank' style='text-decoration:none'>Liujuan Cao<sup>1</sup></a>, &ensp;
    <a href='https://mac.xmu.edu.cn/rrji/' target='_blank' style='text-decoration:none'>Rongrong Ji<sup>1</sup></a>
</div>

<div>
    <sup>1</sup> Key Laboratory of Multimedia Trusted Perception and Efficient Computing, <br> Ministry of Education of China, Xiamen University, China.<br>
    <sup>2</sup> Kunlun Skywork AI.
</div>

<sub></sub>

<p align="center">
    <a href="https://arxiv.org/abs/2410.20084" target="_blank"> <img src='https://img.shields.io/badge/arXiv-2410.20084%20UniVST-red' alt='Paper PDF'></a>
    <a href='https://quanjiansong.github.io/projects/UniVST'><img src='https://img.shields.io/badge/Project_Page-UniVST-green' alt='Project Page'></a>
</p>
</div>

---

## 🔥🔥🔥 News
<pre>
• <strong>2024.10.26</strong>: 🔥 The paper of UniVST has been submitted to <a href="https://arxiv.org/abs/2410.20084">arXiv</a>.
• <strong>2025.01.01</strong>: 🔥 The official code of UniVST has been released.
</pre>

## 🎬 Overview
![overview](assets/overall_framework.png)


## 🔧 Environment
```
# Git clone the repo
git clone https://github.com/QuanjianSong/UniVST.git

# Installation with the requirement.txt
conda create -n UniVST python=3.9
conda activate UniVST
pip install -r requirements.txt

# Or installation with environment.yaml
conda env create -f environment.yaml
```

## 🚀 Start
#### ► 1.Perform inversion for original video
```
python content_ddim_inv.py --content_path ./examples/content/libby \
                            --output_dir ./output
```
Then, you will find the content inversion result in the `./output/content`.

#### ► 2.Perform mask propagation
```
python mask_propagation.py --feature_path ./output/features/libby/inversion_feature_301.pt \
                            --mask_path ./examples/mask/libby.png \
                            --output_dir ./output
```
Then, you will find the mask propagation result in the `./output/mask`.

#### ► 3.Perform inversion for style image
```
python style_ddim_inv.py --style_path ./examples/style/style1.png \
                            --output_dir ./output
```
Then, you will find the style inversion result in the `./output/style`.

#### ► 4.Perform localized video style transfer
```
python video_style_transfer.py --inv_path ./output/content/libby/inversion\
                            --mask_path ./output/mask/libby\
                            --style_path ./output/style/style1/inversion\ 
                            --output_dir ./output
```
Then, you will find the edit result in the `./output/edit`.


## 🎓 Citation
If you find this code helpful for your research, please cite:
```
@article{song2024univst,
  title={UniVST: A Unified Framework for Training-free Localized Video Style Transfer},
  author={Song, Quanjian and Lin, Mingbao and Zhan, Wengyi and Yan, Shuicheng and Cao, Liujuan and Ji, Rongrong},
  journal={arXiv preprint arXiv:2410.20084},
  year={2024}
}
```
