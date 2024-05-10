# emonet-pytorch
EmoNet, as seen in [Kragel et al., 2019 (Science Advances)](https://www.science.org/doi/10.1126/sciadv.aaw4358), but implemented in PyTorch.

## What this is

The original EmoNet was created by fine-tuning a [pre-trained AlexNet object recognition model](https://www.mathworks.com/help/deeplearning/ref/alexnet.html) as implemented in Matlab (need to confirm the version, but probably R2017-something).

This model spec will allow you to spin up a pre-trained instance of EmoNet in PyTorch!

## How it was made

While AlexNet has one specific architecture, it's implemented using slightly different logic in different neural network packages, so converting the model required more than exporting a matrix of weights from an EmoNet instance using the Matlab Deep Learning Toolbox and importing into Python to paste in as the weights of a torchvision AlexNet instance.

I converted the model via the following:

- Exported Matlab-EmoNet to an ONNX file using `exportONNXnetwork()` from the **Matlab** Deep Learning Toolbox
- Imported that ONNX model into **Python** as an ONNX v13 model using `load()` and `version_converter.convert()` from the onnx package
- Converted that python-ONNX model to a PyTorch model instance using onnx2torch v1.5.2 (should work in any version _later_ than this as well)
- With these now PyTorch-formatted model weights, re-inserted these weights into a slightly more "pythonic" AlexNet architecture (to bring the under-the-hood setup closer to the torchvision native AlexNet architecture)

So, the model works in PyTorch now! However, because of the conversion method, the weights are not in an architecture compatible with `torchvision.models.alexnet`. Even though EmoNet is an AlexNet model, you have to import it this way, using my code to create a specific PyTorch architecture, not just by importing the weight matrices.

## How to use it

Right now, it's super lightweight (for better or worse). Download this repo, and you can reference

```python
import torch
from torch import nn
from models import EmoNet

this_emonet = EmoNet()
this_emonet.load_state_dict_from_web()
```

That last command will use PyTorch's built in methods to download the  the PyTorch/numpy-compatible pre-trained weights file from [its OSF repo](https://osf.io/amdju) and then read the weights into your active model instance from that local file. 

Once you've downloaded the weights once, you can reference your local copy instead of downloading them every time:

```python
this_emonet.load_state_dict_from_path(path_to_weights_file)
```

The provided replicate-kragel2019.py example script shows how you can read in a PyTorch-style Dataset containing the Cowen & Keltner (2017) video database that EmoNet was originally trained on, and output its predictions on held-out testing video frames from that stimulus set. (Please note! It currently assumes the videos in the Cowen (2017) folder are sorted into subfolders by their emotion class name.)

### Prerequisite packages

Importing and using the model only requires pytorch (and its dependencies) to be installed. The model was converted with pytorch v1.12.1, so if you are using it in a newer version of pytorch, you are responsible for checking over any differences in implementation for the Conv2d, ReLU, LocalResponseNorm, and MaxPool2d torch.nn modules used in this architecture.
