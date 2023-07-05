# SCGen

SCGen is a project to automatically generate snippet comments for given code snippets with their contexts.
The source code and dataset are opened.

# Requirements

Python 3.7

Pytorch 1.8

NLTK 3.5

# Config

We can change the configuration in the `main.py` by setting the field values of the class `Config`.

# Model Training

Train the BCGen: `python main.py BCGen [main_folder]`

In every training epoch, if the model has the greater performance, this model will be saved in "`[main_folder]`/model/BCGen_`[training_epoch]`".

# Dataset

Our dataset is opened in [Zenodo](https://zenodo.org/record/8112810)


