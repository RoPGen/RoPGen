## Robustness of author attribution
The purpose of this project is to use GradAug method to enhance the robustness of author attribution of PBNN model.

## Datasets
* GCJ C++/JAVA
* GitHub C/Java40

## Project structure

Models and all the code for training/evaluation located in [authorship_pipeline](attribution/authorship_pipeline) directory.
To run experiments:
1 Create configuration file manually (for examples see [configs](attribution/authorship_pipeline/configs) directory).

2 Preprocessing training set data:
2.1 Put source code files in `datasets/datasetName/{author}/{files}`. Make sure files of each author are in a single directory. 
2.2 Run data extraction to mine path-contexts from the source files:
```
java -jar attribution/pathminer/extract-path-contexts.jar snapshot \
    --project datasets/datasetName/ \
    --output processed/datasetName/ \
    --java-parser antlr \
    --maxContexts 1000 --maxH 8 --maxW 3
```
java dataset : maxH=8 and maxW=3
c,c++ dataset : maxH=6 and maxW=2
2.3 Depending on the language, extracted data will be in the `processed/datasetName/{c,cpp,java,py}` folder.

3 Train
3.1 Create a configuration file (e.g., [PbNN](attribution/authorship_pipeline/configs/java40/nn).
3.1.1 Switching training method :
Modify the `training_method` in the configuration file, 
`training_method : GradAug`  indicates use GradAug method to train 
`training_method : original`  indicates use original method to train
3.2 Run `python -m run_classification path/to/config` in `attribution/authorship_pipeline` folder.

4 Test 
4.1 Perform step 2 on the test dataset.
4.2 In order to ensure that the token and path tags of the test set are consistent with those of the training set,run `python merge.py` in `attribution/authorship_pipeline` folder.
4.3 Modify the `source_folder` in the configuration file to the path-contexts path of the test set, `final_test: 1` (default `0` indicates training).
4.4 Run `python -m run_classification path/to/config` in `attribution/authorship_pipeline` folder.