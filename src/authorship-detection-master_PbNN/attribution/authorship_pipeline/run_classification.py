import argparse
import os,glob

import yaml
import pickle
from classifiers.BaseClassifier import ClassificationResult
from classifiers.CaliskanClassifier import CaliskanClassifier
from classifiers.NNClassifier import NNClassifier
from classifiers.RFClassifier import RFClassifier
from classifiers.config import Config
from preprocessing.compute_occurrences import compute_occurrences
from preprocessing.context_split import context_split
from preprocessing.resolve_entities import resolve_entities
from preprocessing.time_split import time_split
from util import ProcessedFolder, ProcessedSnapshotFolder
from pathlib import Path
#g_modelfile="/home/zss/data/project5/authorship-detection-master_epoch_nn_gcj5/attribution/authorship_pipeline/nn_output/grad_3_2mcts_15/"
g_modelfile="/home/zss/data/project5/authorship-detection-master_epoch_nn_gcj5/attribution/authorship_pipeline/output/githubc2/"
def output_filename(input_file):
    if not os.path.exists('output'):
        os.mkdir('output')
    return 'output/' + input_file


def output_file(input_file):
    output_file = output_filename(input_file)
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))
    return open(output_file, 'w')


def main(args):
    '''
    import ptvsd
    ptvsd.enable_attach(address = ('10.188.65.192', 3004))
    ptvsd.wait_for_attach()
    '''
    global g_modelfile
    config = Config.fromyaml(args.config_file)

    if config.mode() == 'snapshot':
        project_folder = ProcessedSnapshotFolder(config.source_folder())
        change_entities = None
        author_occurrences = None
    else:
        project_folder = ProcessedFolder(config.source_folder())
        change_entities = resolve_entities(project_folder)
        author_occurrences, _, _, _ = compute_occurrences(project_folder)

    if config.mode() == 'time':
        change_to_time_bucket = time_split(project_folder, config.time_folds(), uniform_distribution=True)
    else:
        change_to_time_bucket = None

    if config.mode() == 'context':
        context_splits = context_split(project_folder, *config.min_max_count(), *config.min_max_train())
    else:
        context_splits = None

    if config.classifier_type() == 'nn':
        classifier = NNClassifier(config, project_folder, change_entities, change_to_time_bucket,
                                  config.min_max_count(), author_occurrences, context_splits)
    elif config.classifier_type() == 'rf':
        classifier = RFClassifier(config, project_folder, change_entities, change_to_time_bucket,
                                  config.min_max_count(), author_occurrences, context_splits)
    elif config.classifier_type() == 'caliskan':
        classifier = CaliskanClassifier(config, project_folder, change_entities, change_to_time_bucket,
                                        config.min_max_count(), context_splits)
    else:
        raise ValueError('Classifier type should be set in config')

    if config.mode() == 'time':
        fold_indices = [(i, j) for i in range(config.time_folds()) for j in range(i + 1, config.time_folds())]
    elif config.mode() == 'context':
        fold_indices = [i for i in range(len(context_splits))]
    else:
        fold_indices = [0]

    skip_read_write = True if classifier.config.skip_read_write()==1 else False
    isModelExist = False
    train_original_labels = []
    if not skip_read_write:
        # Loading model
        train_original_labels = []
        if Path(g_modelfile+"0.pck").exists():
            isModelExist = True
        if isModelExist:
            if isinstance(classifier.config.test_size(), int):
                fold_indices = list(range(len(glob.glob(g_modelfile+"*.pck"))-1))
            # Loading train_original_labels
            train_original_labels = pickle.load(open(g_modelfile + "train_original_labels.pck", 'rb'))
            for fold_ind in fold_indices:
                modelfile = g_modelfile + str(fold_ind)+".pck"
                print("Loading model --> "+modelfile)
                classifier.models[fold_ind] = pickle.load(open(modelfile, 'rb'))
        
    mean, std, scores, original_labels = classifier.run(fold_indices,train_original_labels)#进入交叉验证
    
    print(original_labels)
    if not isModelExist and not skip_read_write:
        # Save the train_original_labels for correcting the label of the test set
        pickle.dump(original_labels, file=open(g_modelfile + "train_original_labels.pck", 'wb'))
        for fold_ind in fold_indices:
            # save model
            modelfile = g_modelfile + str(fold_ind)+".pck"
            print("Save label --> "+modelfile)
            pickle.dump(classifier.models[fold_ind], file=open(modelfile, 'wb'))
        
    print(f'{mean:.3f}+-{std:.3f}')
    for i, score in enumerate(scores):
        if isinstance(score, ClassificationResult):
            scores[i] = ClassificationResult(
                float(score.accuracy), float(score.macro_precision), float(score.macro_recall),
                score.fold_ind
            )

    yaml.dump({
        'mean': mean,
        'std': std,
        'scores': scores
    }, output_file(args.config_file), default_flow_style=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='config_file', type=str, help='Configuration file in YAML format')
    args = parser.parse_args()
    main(args)
