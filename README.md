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

### Prerequisite packages


