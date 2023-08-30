# emonet-pytorch
EmoNet, as seen in Kragel et al., 2019 (Science Advances), but implemented in PyTorch.

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

So, the model works in PyTorch now! However, because of the conversion method, the weights are not in a format compatible with `torchvision.models.alexnet`. Even though EmoNet is an AlexNet model, you have to import it this way, using my code to create a specific PyTorch architecture, not just by importing the weight matrices.

## How to use it

Right now, it's super lightweight (for better or worse). Download this repo (or just the models.py file), and you can reference

```python
import torch
from torch import nn
from models import EmoNet

this_emonet = EmoNet()
this_emonet.load_state_dict_from_path(path_to_weights_file)
```

That last command will use the `load_state_dict()` method to read in the PyTorch/numpy-compatible pre-trained weights file, saved as a .pt file. Right now, you need to email me to get a copy of the .pt weights, but we will upload them asap and update the `load_state_dict_from_path()` method to accept a URL, so you can pull the weights directly off the web and into your model instance.

### Prerequisite packages

Importing and using the model only requires pytorch (and its dependencies) to be installed. The model was converted with pytorch v1.12.1, so if you are using it in a newer version of pytorch, you are responsible for checking over any differences in implementation for the Conv2d, ReLU, LocalResponseNorm, and MaxPool2d torch.nn modules used in this architecture.
