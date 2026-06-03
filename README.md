# LILogic Net - Learnable Interconnect Logic Network

This repository includes the implementation of LILogic Net framework presented in ECV @ CVPR 2026 paper "LILogic Net: Compact Logic Gate Networks with Learnable Connectivity for Efficient Hardware Deployment" ([ArXiv](https://arxiv.org/pdf/2511.12340)).

## Prerequisites

This repo uses the *uv* package manager, see the [uv repo](https://github.com/astral-sh/uv/) for installation instructions or simply install *uv* via pip:

```bash
pip install uv
```

## How to run

Run the script to train LGN architecture with 2 layers of 8000 gates each and learnable connections between them.
```bash
uv run lilogic_net.py
```
It will produce small 2-layer Logic Gate Network with an accuracy of **97.9%** on the MNIST dataset.

## Config

You can override almost any of the training parameters by supplying environment variables!

Put variables in the command line just before calling our script, for example:
```bash
GATE_ARCHITECTURE="[4000, 2550]" INTERCONNECT_ARCHITECTURE="[[],[-1]]" C_INIT="NORMAL" IMG_WIDTH=16 IMG_CROP=22 LEARNING_RATE=0.03 EPOCHS=100 SEED=230646   uv run lilogic_net.py
```

It will produce ultra tiny 15K parameters 2-layer Logic Gate Network that is still capable of a remarkable **97.6%** accuracy on the MNIST dataset.

The following is the subset of the most interesting and useful configuration variables:
- `DATASET`, `LEARNING_RATE`, `EPOCHS`,
- `GATE_ARCHITECTURE`, `INTERCONNECT_ARCHITECTURE`,
- `SCALE_LOGITS` and `MANUAL_GAIN` see below for more defails.

### Dataset and Image resolution

- **`DATASET`** currently supports the following settings: `MNIST` (default), `FashionMNIST`, `KMNIST`, `QMNIST`, `EMNIST`, `CIFAR10`.

- **`BINARIZE_IMAGE_TRESHOLD`** supports an array of thresholds for thermometer encoding of the input pixels. For example `BINARIZE_IMAGE_TRESHOLD="[0.25, 0.5, 0.75]"` will produce 3 inputs for each input pixel. Respecitively pixel of value `0.3` will result in a binary vector of `1,0,0`, value of `0.6` will become `1,1,0` and so on. It is useful for the datasets like **FashionMNIST** that contain grayscale natural images.

- **`RGB_TO`** supports `MONO` (default) and `LAB` values. By default input images are converted to grayscale aka monochrome, but specifying `RGB_TO="LAB"` will convert RGB pixels into a CIE Lab perceptual color space before passing into thermometer encoding controller by `BINARIZE_IMAGE_TRESHOLD`. It is useful for the datasets like **CIFAR10** that contain colored natural images. [CIE Lab](https://en.wikipedia.org/wiki/Oklab_color_space) perceptual color space should be more suitable encoding for Machine Learning than RGB.

- **`IMG_WIDTH`** scales input image to a particular resolution, by default it takes the value of 28 for MNIST datasets and 32 for CIFAR.

- **`IMG_CROP`** crops input image, by default it is equal to `IMG_WIDTH`

To train on CIFAR10 dataset while converting every input pixel into 9 binary element vector, run:
```bash
DATASET="CIFAR10" BINARIZE_IMAGE_TRESHOLD="[0.25, 0.5, 0.75]" RGB_TO="LAB"  C_INIT="NORMAL"  uv run lilogic_net.py
```

### Network architecture
- **`GATE_ARCHITECTURE`** specifies a number of logic gates for each layer of the network. `GATE_ARCHITECTURE="[4000, 2550]"` creates a 2 layer network with 4000 gates for the first layer and 2550 for the last one,  `GATE_ARCHITECTURE="[8000, 8000, 8000, 8000]"` creates a 4 layer architecture with 8000 gates each.

- **`INTERCONNECT_ARCHITECTURE`** specifies a type (and parameters) for interconnect between the layers where the first parameter specifies interconnect between the input and the first layer:
  - `[]` trainable interconnect
  - `[k, 5]` trainable "TopK" interconnect which will allow each gate to select one of the 5 random connections chosen at the beginning of the training. NOTE that using value higher than 8 can significantly slow down the training.
  - `[-1]` fixed (non trainable) random interconnect
 
- **`C_INIT`** accepts either `DIRAC` (default) for sparse initialisation or `NORMAL` for gaussian initialisaion for the trainable interconnect. Natural images and tiny shallow networks might prefer `NORMAL` initialisation.

### Logits
- **`MANUAL_GAIN`** specify TAU value. Default is 10.

To quickly, _just in 10 epochs_ train small, but capable **48%** accuracy network on CIFAR10 dataset, run:
```bash
DATASET="CIFAR10" IMG_WIDTH=22 BINARIZE_IMAGE_THRESHOLD="[0.25, 0.5, 0.75]" RGB_TO="LAB"   GATE_ARCHITECTURE="[8000,16000]" INTERCONNECT_ARCHITECTURE="[],[-1]" C_INIT="NORMAL"   LEARNING_RATE=0.075 EPOCHS=10  uv run  lilogic_net.py
```

### Training parameters
- **`SEED`** if not specified, random seed is picked,
- **`BATCH_SIZE`** default is `256`,
- **`EPOCHS`** default is `30`,
- **`LEARNING_RATE`** default is `0.075`.

## Configuration file

To manage configuration settings, you can use optional `.env` file. A sample MNIST configuration is available in `.env.mnist`.  To set up, copy `.env.mnist` to `.env` and update it as needed.
For security reasons, the `.env` file is excluded from version control to prevent accidental exposure of sensitive data like API keys.

## Experiments from paper

We present a few experiments which are included in the paper.

### MNIST

| Dataset       | Architecture | τ  | Test Accuracy [%] |
|---------------|--------------|----|-------------------|
| LILogic Net-S | 1L-4K        | 5  | 97.96 ± 0.11      |
| LILogic Net-M | 1L-8K        | 10 | 98.45 ± 0.07      |
| LILogic Net-L | 2Top32-16K   | 10 | 98.95 ± 0.09      |

Use `.env.mnist` file with baseline MNIST settings and the following commands:

```bash
LOG_TAG=Lx4K_mnist GATE_ARCHITECTURE=[4000] INTERCONNECT_ARCHITECTURE="[[]]" MANUAL_GAIN=5 uv run lilogic_net.py
LOG_TAG=Lx8K_mnist GATE_ARCHITECTURE=[8000] INTERCONNECT_ARCHITECTURE="[[]]" MANUAL_GAIN=10 uv run lilogic_net.py
LOG_TAG=2Top32x16K_mnist GATE_ARCHITECTURE=[16000,16000] INTERCONNECT_ARCHITECTURE="[['k',32],['k',32]]" MANUAL_GAIN=10 uv run lilogic_net.py
```

### FashionMNIST

| Dataset       | Architecture | τ  | Test Accuracy [%] |
|---------------|--------------|----|-------------------|
| LILogic Net-S | 1L-8K        | 15 | 89.95 ± 0.06      |
| LILogic Net-M | 2Top32-32K   | 25 | 90.26 ± 0.11      |
| LILogic Net-L | 2Top32-64K   | 30 | 90.61 ± 0.09      |

Use `.env.fmnist` file with baseline FashionMNIST settings and the following commands:

```bash
LOG_TAG=Lx8K_fmnist GATE_ARCHITECTURE=[8000] INTERCONNECT_ARCHITECTURE="[[]]" MANUAL_GAIN=15 uv run lilogic_net.py
LOG_TAG=2Top32x32K_fmnist GATE_ARCHITECTURE=[32000,32000] INTERCONNECT_ARCHITECTURE="[['k',32],['k',32]]" MANUAL_GAIN=25 uv run lilogic_net.py
LOG_TAG=2Top32x64K_fmnist GATE_ARCHITECTURE=[64000,64000] INTERCONNECT_ARCHITECTURE="[['k',32],['k',32]]" MANUAL_GAIN=25 uv run lilogic_net.py
```

### CIFAR10

| Dataset       | Architecture | τ  | Test Accuracy [%] |
|---------------|--------------|----|-------------------|
| LILogic Net-S | 1L-8K        | 20 | 55.11 ± 0.17      |
| LILogic Net-M | 1L-64K       | 90 | 57.66 ± 0.17      |
| LILogic Net-L | 2Top32-128K  | 100| 60.98 ± 0.19      |

Use `.env.cifar10` file with baseline FashionMNIST settings and the following commands:

```bash
LOG_TAG=Lx8K_cifar10 GATE_ARCHITECTURE=[8000] INTERCONNECT_ARCHITECTURE="[[]]" MANUAL_GAIN=20 uv run lilogic_net.py
LOG_TAG=Lx64K_cifar10 GATE_ARCHITECTURE=[64000] INTERCONNECT_ARCHITECTURE="[[]]" MANUAL_GAIN=90 uv run lilogic_net.py
LOG_TAG=2Top32x128K_cifar10 GATE_ARCHITECTURE=[128000,128000] INTERCONNECT_ARCHITECTURE="[['k',32],['k',32]]" MANUAL_GAIN=100 uv run lilogic_net.py
```


## Citing

```bash
@article{fojcik2025lilogic,
  title={LILogic Net: Compact Logic Gate Networks with Learnable Connectivity for Efficient Hardware Deployment},
  author={Fojcik, Katarzyna and Zioma, Renaldas and Armaitis, Jogundas},
  journal={arXiv preprint arXiv:2511.12340},
  year={2025}
}
```

## Licence

LILogic Net is released under the Transcendent Logic Non-Commercial License. See [LICENCE](./LICENSE) for additional details about it.

