# Introduction
This README file instructs you on how to train a vanilla (non-robust) CCS18 (DL-CAIS) model on datasets other than the original C++ one.

For guidance on how to train a GradAug CCS18 model, see `gradaug_readme.md`.

For guidance on how to test a model (vanilla or GradAug), see the "Testing" section in `gradaug_readme.md`.

## Project Structure
Four Python scripts matter the most in our training. They are:
- `<project root>/src/PyProject/evaluations/learning/rnn_css/vanilla_train.py`
- `<project root>/src/PyProject/evaluations/learning/rnn_css/MySequence.py`
- `<project root>/src/PyProject/classification/utils_learning_rnn.py`
- `<project root>/src/PyProject/classification/LearnSetups/LearnSetupRNNRF.py`

Please refer to the README of the original model for the rest of the files in the project.

Now we explain briefly the purposes of these four scripts.

**vanilla_train.py** - this is where we set up hyperparameters for the model, and where we kick-start the training

**MySequence.py** - this is where we supply a dataset every epoch for the model to learn

**utils_learning_rnn.py** - the CCS18 model architecture is defined here

**LearnSetupRNNRF.py** - class that contains some metadata about the trained model, and will be saved together with the model itself

## Requirements
- Python 3.x
- Tensorflow 1.x (oops, maybe 2.x will work if you change all `keras` imports to `tensorflow.keras`, but we haven't tried)

See `<project root>/src/PyProject/requirements.txt` for other dependencies required by the original model.

# Data Preprocessing
**This section is the same as the one in `gradaug_readme.md`, so skip this one now if you've read that one.**

Your training set and testing set must follow a directory structure which will be described below.

**In a nutshell: one directory for each author, and all programs into their respective author's directory.**

Assuming your dataset path is `<dataset>`, each author should have their own directory under `<dataset>`. For example, if there are three authors A, B, and C, then you should have:
```
<dataset>/A/
<dataset>/B/
<dataset>/C/
```

All the programs belonging to a certain author should be placed directly under that author's directory. By "directly" is meant that no intermediate directories are allowed. For example, if B authored two programs 1.cpp and 2.cpp, then you should have:
```
<dataset>/B/1.cpp
<dataset>/B/2.cpp
```

# Training
**This section is different from the one in `gradaug_readme.md`, so read it carefully.**

Starting the training is as easy as running the following:

```
python vanilla_train.py
```

**However**, you must do a few things before you can get started. There are a few places you must change in the scripts. These places are marked with "`FIXME #<number>`" in several comments in `vanilla_train.py` **AND** `MySequence.py`. We will explain them one by one. Some changes are optional, which will be noted below.

## Dataset
- (in `vanilla_train.py`) `FIXME #1` - change this to the full path to your project root directory (i.e., `code-imitator`).
- (in `vanilla_train.py`) `FIXME #2` - change this to the full path to your training set.
- (in `vanilla_train.py`) `FIXME #3` - change this to the full path to your testing set. (note: in `vanilla_train.py` there is actually no testing, but a path to the testing set is required; see the comments in the script for why)
- (in `vanilla_train.py`) `FIXME #4` - change this to the number of authors in your training set.

## Model
- (in `MySequence.py`) `FIXME #9` - change this to the number of authors in your training set.
- (in `MySequence.py`) `FIXME #10` - change this to the number of authors in your testing set.
- (Optional, in `vanilla_train.py`) `FIXME #5` - change this to the number of training epochs.
- (Optional, in `vanilla_train.py`) `FIXME #6` - write your own grid search code if you want.
- (Optional, in `vanilla_train.py`) `FIXME #8` - change this to the batch size you want. Note there are two places marked `FIXME #8`; they should be changed together
- (Optional, in `MySequence.py`) `FIXME #11` - change this to how frequently you want a checkpoint model to be saved (see comments for detail).
- (Optional, in `MySequence.py`) `FIXME #12` - change this if you want a model without an RF classifier (be careful as this changes the model architecture).
- (Optional, in `MySequence.py`) `FIXME #13` - change this to the path to save your model into.

## GPU
- (Optional, in `vanilla_train.py`) `FIXME #7` - change this to the GPU you want to use.

Now that you have made the necessary changes, training is just one command away:

```
python vanilla_train.py
```
