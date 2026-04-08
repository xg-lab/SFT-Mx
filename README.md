
<h1 align="center"><strong>SimpleFold-Turbo: Folding Proteins is <i>Way</i> Faster than You Think</strong></h1>


<div align="center">

This github repository accompanies the research paper [*SimpleFold-Turbo: Adaptive caching yields 14-fold acceleration in flow-matching protein structure generation*]() (pending, bioRxiv 2026).

*Geoffrey Taghon, NIST*

[[`Paper`](publication/simplefold_turbo_2026.pdf)]  [[`BibTex`](#citation)]

<img src="assets/intro.png" width="750">

</div>


## Introduction

We introduce SimpleFold-Turbo, an efficient adaptive caching fork of [SimpleFold](https://github.com/apple/ml-simplefold). SimpleFold does not rely on expensive modules like triangle attention or pair representation biases, and is trained via a generative flow-matching objective. In search of further speed optimization, we apply adaptive timestep caching following the example of [TeaCache](https://github.com/ali-vilab/TeaCache) video diffusion acceleration to achieve an order-of-magnitude speedup over original SimpleFold, and a significant speedup over CUDA-based AlphaFold3. SimpleFold-Turbo is fully open source, requires no internet connection for operation, and is highly suitable for ensemble predictions, as individual structures are produced in milliseconds to seconds on consumer hardware (Apple M2 Max). Caching and relative speedup is only dependent on input sequence length, **not** biophysical or geometric properties of the determined fold.

</div>


## Installation

To install `simplefold-turbo` package from github repository, run
```
git clone https://github.com/usnistgov/simplefold-turbo.git
cd simplefold-turbo
conda create -n sft python=3.10
conda activate sft
python -m pip install -U pip build; pip install -e .
```
If you want to use the MLX backend on Apple silicon (**Highly Recommended**): 
```
pip install -U mlx
pip install git+https://github.com/facebookresearch/esm.git
```

## Example 

We provide a jupyter notebook [`sample.ipynb`](sample.ipynb) to predict protein structures from example protein sequences. 

## Inference

Once you have the `simplefold-turbo` package installed, you can predict protein structures from target fasta file(s) via the following command line invocation. We provide support for both [PyTorch](https://pytorch.org/) and [MLX](https://mlx-framework.org/) (recommended for Apple hardware) backends in inference. 
```
sft \
    --simplefold_model simplefold_100M \  # specify base folding model in simplefold_100M/360M/700M/1.1B/1.6B/3B
    --num_steps 500 --tau 0.01 \          # specify inference setting
    --nsample_per_protein 1 \             # number of generated conformers per target
    --plddt \                             # output pLDDT
    --fasta_path [FASTA_PATH] \           # path to the target fasta directory or file
    --output_dir [OUTPUT_DIR] \           # path to the output directory
    --backend [mlx, torch] \              # choose from MLX and PyTorch for inference backend
    --teacache 0.1                        # TeaCache threshold (0.0 = off, 0.1 = default)
```

## Train

You can also train or tune SimpleFold-Turbo on your end. Instructions are the same as for [SimpleFold](https://github.com/apple/ml-simplefold?tab=readme-ov-file#train). 


## Citation
If you found this code useful, please cite the following papers:
```
@article{simplefold-turbo,
  title={SimpleFold-Turbo: Adaptive caching yields 14x acceleration in flow-matching protein structure generation},
  author={Taghon, Geoffrey},
  journal={bioRxiv preprint bioRxiv:pending},
  year={2026}
}
@article{simplefold,
  title={SimpleFold: Folding Proteins is Simpler than You Think},
  author={Wang, Yuyang and Lu, Jiarui and Jaitly, Navdeep and Susskind, Josh and Bautista, Miguel Angel},
  journal={arXiv preprint arXiv:2509.18480},
  year={2025}
}
```

## Dataset
Complete raw data, including structural models generated during this work, is published at [Zenodo](TBD).

## Acknowledgements
This codebase was built using multiple opensource contributions, please see [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS) for more details. 

## License
Please check out the repository [LICENSE](LICENSE) before using the provided code and
[LICENSE_MODEL](LICENSE_MODEL) for the released models.
