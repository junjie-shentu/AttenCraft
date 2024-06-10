# AttenCraft: Attention-guided Disentanglement of Multiple Concepts for Text-to-Image Customization

This is the implementation of the paper "AttenCraft: Attention-guided Disentanglement of Multiple Concepts for Text-to-Image Customization". [Paper Link](https://arxiv.org/abs/2405.17965)

## Getting Started

Intsall environment:
```
conda create --name attencraft --file environment.yml
conda activate attencraft
```

## Training AttenCraft
```
bash train.sh
```

Note that the `--output_dir` flag specifies the output directory where the checkpoints will be saved, and should contain 'wkwv' since this will be used in the inference script.

## Inference
Input the chackpoint path, output path, and the text ptompt for the image generation in the `inference.py` file and run the python script.

## Citation
If you find this work helpful, please consider citing the following BibTeX entry:
```
@article{shentu2024attencraft,
  title={AttenCraft: Attention-guided Disentanglement of Multiple Concepts for Text-to-Image Customization},
  author={Shentu, Junjie and Watson, Matthew and Moubayed, Noura Al},
  journal={arXiv preprint arXiv:2405.17965},
  year={2024}
}
```