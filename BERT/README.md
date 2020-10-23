# Spoken language understanding

run_slu.py used for SLU task, it is based on the hugging face script [`run_ner.py`](https://github.com/huggingface/transformers/blob/master/examples/ner/run_ner.py) for Pytorch.
This example fine-tune Bert (Camembert or Flaubert) base models on MEDIA task.

##Experiments 
 BERT models can be downloaded from [here](https://huggingface.co/models?search=camembert)
 The bash scripts are used to run experiments on MEDIA task, using one of the french BERT models.  




 ## Installation

This repo is tested on Python 3.6+, PyTorch 1.0.0+ and TensorFlow 2.0.0-rc1

You should install 🤗 Transformers in a [virtual environment](https://docs.python.org/3/library/venv.html). If you're unfamiliar with Python virtual environments, check out the
 [user guide](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/).

Create a virtual environment with the version of Python you're going to use and activate it.

Now, if you want to use 🤗 Transformers, you can install it with pip. If you'd like to play with the examples, you must install it from source.

### With pip

First you need to install one of, or both, TensorFlow 2.0 and PyTorch.
Please refer to [TensorFlow installation page](https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available) and/or [PyTorch installation page](https://pytorch.org/g
et-started/locally/#start-locally) regarding the specific install command for your platform.

When TensorFlow 2.0 and/or PyTorch has been installed, 🤗 Transformers can be installed using pip as follows:

```bash
pip install transformers
```

### From source

Here also, you first need to install one of, or both, TensorFlow 2.0 and PyTorch.
Please refer to [TensorFlow installation page](https://www.tensorflow.org/install/pip#tensorflow-2.0-rc-is-available) and/or [PyTorch installation page](https://pytorch.org/g
et-started/locally/#start-locally) regarding the specific install command for your platform.

When TensorFlow 2.0 and/or PyTorch has been installed, you can install from source by cloning the repository and running:

```bash
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
```

When you update the repository, you should upgrade the transformers installation and its dependencies as follows:

```bash
git pull
pip install --upgrade .
```
