## Zero-Inflated Gamma probabilistic model

This code implements the Zero-Inflated Gamma encoding and decoding model for post-deconvolved calcium imaging traces.

## Tutorial

A Jupyter Notebook tutorial is available in the code folder. There're more examples in the examples in paper folder.

## Installation

The code is written in Python 3.6. In addition to standard scientific Python libraries (numpy, scipy, matplotlib), the code expects: Tensorflow (>=1.9.0)

To download this code, run `git clone https://github.com/zhd96/zig.git`

## Data

The real datasets analyzed in the paper<sup>1</sup> can be downloaded at https://drive.google.com/drive/folders/1roTjSoEpBGNjJEaL89c_16svp6vYOp0_?usp=sharing. 

There is a sample simulated data in the data folder. The code to generate the simulated datasets analyzed in the paper<sup>1</sup> is in the code folder.


## Reference

If you use this code, please cite the paper:

1. Wei, X.X., Zhou, D., Grosmark, A., Ajabi, Z., Sparks, F., Zhou, P., Brandon, M., Losonczy, A. and Paninski, L., 2019. A zero-inflated gamma model for post-deconvolved calcium imaging traces. bioRxiv, p.637652. https://www.biorxiv.org/content/10.1101/637652v1
