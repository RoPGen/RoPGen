# Introduction
This README file instructs you on how to train a robust CCS18 (DL-CAIS) model with GradAug (i.e., RoPGen framework) and how to test a model.

For guidance on how to train a vanilla CCS18 model on datasets other than GCJ C++, see `vanilla_readme.md`.

## Project Structure
Five Python scripts play the main part in our GradAug training and testing. They are:
- `<project root>/src/PyProject/evaluations/learning/rnn_css/gradaug_train.py`
- `<project root>/src/PyProject/evaluations/learning/rnn_css/test.py`
- `<project root>/src/PyProject/evaluations/learning/rnn_css/test_t.py`
- `<project root>/src/PyProject/classification/utils_learning_rnn_ga.py`
- `<project root>/src/PyProject/classification/LearnSetups/LearnSetupRNNRF_ga.py`

Please refer to the README of the original model for the rest of the files in the project.

Now we explain briefly the purposes of these five scripts.

**gradaug_train.py** - pretty straightforward, as this is where the training takes place

**test.py** - use this to test the models you have trained (for original samples and untargeted attack samples)

**test_t.py** - use this to test the models you have trained (for targeted attack samples)

**utils_learning_rnn_ga.py** - the CCS18 model architecture is defined here, and this is where we construct the subnet model from the fullnet one (for each subnet width, we construct a new subnet model, copy the weights from fullnet and slice a part of them, which will then be used as subnet weights. In this way, the weights are shared between full- and subnet)

**LearnSetupRNNRF_ga.py** - class that contains some metadata about the trained model, and will be saved together with the model itself (basically the same as original except we adapted it for TF 2.x)

## Requirements
- Python 3.x
- Tensorflow 2.x (in our case, 2.6.0)

See `<project root>/src/PyProject/requirements.txt` for other dependencies required by the original model.

# Data Preprocessing
**This section is the same as the one in `vanilla_readme.md` EXCEPT FOR THE LAST PARAGRAPH, so skip to the last paragraph of this section now if you've read that one.**

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

**Finally, it is recommended** that you name the programs in the dataset in a way that you can distinguish programmatically between samples used in the fullnet and those used in the subnet, because we'll extract features from `<dataset>` as a whole, and then separate them into fullnet and subnet parts.

# Training
**This section is different from the one in `vanilla_readme.md`, so read it carefully.**

Starting the training is as easy as running the following:

```
python gradaug_train.py
```

**However**, you must change a few places in the script before you can get started. These places are marked with "`FIXME #<number>`" in several comments in `gradaug_train.py`. We will explain them one by one. Some changes are optional, which will be noted below.

## Dataset
- `FIXME #3` - change this to the full path to your project root directory (i.e., `code-imitator`).
- `FIXME #4` - change this to the full path to your training set.
- `FIXME #5` - change this to the full path to your testing set. (note: in `gradaug_train.py` there is actually no testing, but a path to the testing set is required; see the comments in the script for why)
- `FIXME #6` - change this to the number of authors in your training set.
- `FIXME #7` - change this to the number of authors in your testing set.
- `FIXME #8` - change this to the number of authors in your training set.
- (Required only if you're not using our adversarial C++ dataset) `FIXME #1` - this is where we split the training set into two parts, one for fullnet and the other for subnet, as mentioned in the last paragraph in the "Data Preprocessing" section above. If you're using our adversarial dataset, but not the GCJ C++ one, you only need to uncomment some code as instructed in the FIXME comment. If you're NOT using our dataset, you should change this part of code so that in the `if dataset_to_return == 'aug3':` branch, the subnet samples are skipped, and in other branches, the fullnet samples and other irrelevant samples (e.g., samples to be fed into another subnet, see also next FIXME comment) are skipped. By the current implementation, we distinguish full- and subnet samples by their filenames.
- (Required only if you're not using our adversarial C++ dataset) `FIXME #14` - If you're using our adversarial dataset, but not the GCJ C++ one, you only need to comment and uncomment some code as instructed in the FIXME comment. If you're NOT using our dataset, change this part of code as needed to set what samples will be fed into which subnet. Different subnets can receive different samples in the dataset as input, e.g., one for MCTS untargeted adversarial examples and another for MCTS targeted adversarial examples. 

## Model
- (Optional) `FIXME #8` - change this to the batch size you want.
- (Optional) `FIXME #9` - change this to the number of training epochs.
- (Optional) `FIXME #10` - write your own grid search code if you want.
- (Optional) `FIXME #11` - change this to the path to save your model into.
- (Optional) `FIXME #12` - change this to how frequently you want a checkpoint model to be saved (see comments for detail).
- (Optional) `FIXME #13` - change this if you want a model without an RF classifier (be careful as this changes the model architecture).

## GPU
- (Optional) `FIXME #2` - change this to the GPU you want to use.

Now that you have made the necessary changes, training is just one command away:

```
python gradaug_train.py
```

# Testing
With testing, things get a little more complicated, but it basically boils down to seven steps:
1. Determine which testing script should be used.
2. Specify the path to the testing set.
3. Specify the number of authors in the testing set.
4. (if this is the first time you're testing a model trained on this particular set of authors) Create a Python list that associates the authors to their label number as assigned during training. For example, if three authors A, C, and B are assigned label numbers 0, 2, and 1 during training, in this step you create a list `['A', 'B', 'C']`.
5. (for `test.py`) Make up your mind whether or not you want to generate a list of programs correctly classified by the model, and whether or not you want to exclude misclassified programs (based on a previously generated list) from accuracy computation.
6. Change how the script extracts source and target author name from the samples' filename.
7. Run the script and pass the arguments.

## Step 1
If you're testing on original samples or untargeted attack samples, use `test.py`. If you're testing on targeted attack samples, use `test_t.py`.

## Step 2
First, change the project root directory at the very beginning of `test(_t).py` (marked with `FIXME #1`). Then, you need only change the variable `testattackpath` in `test(_t).py` (marked with `FIXME #2`) to point to the testing set directory.

## Step 3
You need only change the parameter `noprogrammers` in `test(_t).py` (marked with `FIXME #5`) to the number of authors in the testing set.

## Step 4
As mentioned above, this step is only needed if this is the first time you're testing a model trained on this particular set of authors. So if you've done this step for authors A, B, and C, then next time you won't need to repeat it for a model trained on these same authors.

In `test(_t).py`, there are already several lists named `authors` (marked with `FIXME #4`) and one of them will be chosen for use based on what you pass as the third argument to the script. If you're testing on one of our datasets, you can now skip this step as we've already done it. If not, you should add a new `elif` branch for your set of authors.

How do you get this list for a particular set of authors? As stated above, this list associates the authors to their label number as assigned during training. The authors in the training set can be accessed by calling `getauthors()` method on your model object's `data_final_train` attribute. Similarly, the label numbers can be accessed via the `getlabels()` method in that attribute. The authors and label numbers you obtain in this way correspond one by one to each other, so the first author's label number is just the first item in the label numbers list, and so on. The problem is the label numbers are out of order, so they can be like `[3, 6, 7, 10, 2, ...]`. You'll have to sort them yourself, and at the same time sort the authors in the same order. For example, you obtain the authors `['A', 'C', 'B']`, and the label numbers `[0, 2, 1]`, you should sort these two lists and get `['A', 'B', 'C']` (the other list is useless now), which will be the final `authors` list you need in `test(_t).py`.

## Step 5
In testing untargeted attack samples, we exclude those samples whose originals were misclassified in the beginning. In order to do this, a list of programs correctly classified must first be generated, and then used to filter misclassified ones out from accuracy computation.

In this step, you should decide whether you want to generate the list (if you're testing on the original testing set), or you want to exclude misclassified samples (if you're testing on the untargeted attack samples set), or neither.

## Step 6
We extract the names of source author and target author from samples' filenames. Since different datasets name the samples differently, this must be taken care of as you switch between datasets.

Search for `FIXME #6` in `test(_t).py` and you'll see the name extraction part. Change this according to your dataset. For example, in the original GCJ C++ dataset, samples are named `<round id>_<challenge id>_<source author>.cpp`, and you can extract `src_author` by splitting the filename by the underscore `_` and then taking the third part and removing `.cpp`.

## Step 7
If you're testing a GradAug model, you should first replace `<project root>/src/PyProject/classification/LearnSetups/LearnSetupRNNRF.py` with `LearnSetupRNNRF_ga.py` under the same directory. `LearnSetupRNNRF.py` was written based on TF 1.x and may not be compatible with TF 2.x. Remember to back up `LearnSetupRNNRF.py` in case you need it again.

If you're testing a vanilla or data-augmented model, use the original `LearnSetupRNNRF.py` (without `_ga`).

We now describe the usage of `test(_t).py`. These scripts should be run with the following command:

```
python test.py <model_path> <keras_model_path> [<authorlist_name>] [-g/-f]
```

```
python test_t.py <model_path> <keras_model_path> [<authorlist_name>]
```

`<model_path>` and `<keras_model_path>` are self-explanatory.

`<authorlist_name>` corresponds to Step 4, where a list of authors is obtained, hardcoded into the script, and given a name. We already have several lists ready for our datasets. They are:

- `cpp` (GCJ-C++)
- `gcj` (GCJ-Java)
- `java40` (GitHub-Java)
- `githubc67` (GitHub-C)

Pass one of the above names as the third argument if you're testing on one of our datasets. Otherwise, you have your own dataset, and you should pass the name you've given it in the code.

The last argument, `-g/-f` corresponds to Step 5. Pass `-g` if you want to generate a list of programs correctly classified (which will be stored directly under the testing set and named `test.txt`). Pass `-f` if you want to exclude misclassified samples based on a previously generated list. Note that if you pass `-f`, you must place the list generated beforehand directly under your testing set and name it `test.txt`. The exclusion is done inside the `for item in testset_list:` loops (in two places). If you're testing on your own dataset using `test.py`, you might want to take a look at the `if` statements inside the loops (marked with `FIXME #7`), and change how it excludes samples to suit your dataset.