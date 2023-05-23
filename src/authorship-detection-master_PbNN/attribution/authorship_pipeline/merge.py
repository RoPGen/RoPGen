import pandas as pd
import numpy as np
from tqdm import tqdm


origin_path = "/home/zss/data/project5/authorship-detection-master_epoch_nn_gcj5/processed/grad/githubc2/test_untar/c/"
new_path = "/home/zss/data/project5/authorship-detection-master_epoch_nn_gcj5/processed/grad/githubc2/test_untar_20/c/"
out_path = "/home/zss/data/project5/authorship-detection-master_epoch_nn_gcj5/processed/grad/githubc2/test_untar_20/c/"


# origin_path = "/home/zss/data/project5/authorship-detection-master_epoch_nn_gcj5/processed/grad/java401/train_aug2and3/java/"
# new_path = "/home/zss/data/project5/nn/proc/java40/1/test_untar_pai/java/"
# out_path = "/home/zss/data/project5/nn/proc/all_dataset_result/java/"
tokens_file1= origin_path + "tokens.csv"
node_types_file1=origin_path + "node_types.csv"
paths_file1=origin_path+"paths.csv"

tokens_file2=new_path+"tokens.csv"
node_types_file2=new_path+"node_types.csv"
paths_file2=new_path+"paths.csv"
path_tokens_file2=new_path+"path_tokens.csv"
path_contexts_file2=new_path+"path_contexts.csv"


def _load_path_contexts_files(path_contexts_file: str):
    '''
    Reads the path contexts file and returns the labels, start tokens, paths and end tokens.
    
    :param path_contexts_file: The file containing the path contexts
    :type path_contexts_file: str
    :return: labels, start_tokens,path,end_tokens
    '''
    raw_data = [line.strip().split(' ', 1) for line in open(path_contexts_file, 'r').readlines()]

    labels = [d[0] for d in raw_data]
    raw_contexts = [d[1] if len(d) == 2 else "1,1,1" for d in raw_data]
    start_tokens = []
    path=[]
    end_tokens = []
    for context in raw_contexts:
        tmp_list = context.split(' ')
        tmp_start = []
        tmp_path = []
        tmp_end = []
        for tmp_str in tmp_list:
            a = tmp_str.split(',')
            tmp_start.append(a[0])
            tmp_path.append(a[1])
            tmp_end.append(a[2])
        start_tokens.append(tmp_start)
        path.append(tmp_path)
        end_tokens.append(tmp_end)

    return labels, start_tokens,path,end_tokens

def _load_rf_contexts_file(rf_contexts_file: str):
    '''
    It takes a file with the following format:
    
    :param rf_contexts_file: The file containing the contexts for each label
    :type rf_contexts_file: str
    :return: The labels, tokens, and paths are being returned.
    '''
    raw_data = [line.strip().split(' ', 1) for line in open(rf_contexts_file, 'r').readlines()]
    labels = []
    tokens = []
    paths = []

    for label, contexts in tqdm(raw_data):
        labels.append(label)
        local_tokens = []
        local_paths = []
        for context in contexts.split():
            t, val = context.split(',')
            if t == 'token':
                local_tokens.append(int(val))
            else:
                local_paths.append(int(val))
        tokens.append(local_tokens)
        paths.append(local_paths)

    return labels, tokens, paths

def _load_paths(paths_file: str):
    paths = _load_stub(paths_file, 'path')
    return _series_to_ndarray(paths)

def _load_node_types(node_types_file: str):
    '''
    Load the node types from the given file and return them as a numpy array.
    
    :param node_types_file: The path to the node_types.csv file
    :type node_types_file: str
    :return: A numpy array of node types.
    '''
    node_types = _load_stub(node_types_file, 'node_type')
    return _series_to_ndarray(node_types)

def _load_tokens(tokens_file: str):
    '''
    Loads the tokens from the given file and returns them as a numpy array.
    
    :param tokens_file: The path to the file containing the tokens
    :type tokens_file: str
    :return: A numpy array of the tokens.
    '''
    tokens = _load_stub(tokens_file, 'token')
    return _series_to_ndarray(tokens)

def _load_stub(filename: str, col_name: str):
    df = pd.read_csv(filename, sep=',', lineterminator='\n', quoting=3)
    df = df.set_index('id')
    return df[col_name]

def _series_to_ndarray(series: pd.Series) -> list:
    converted_values = np.empty(max(series.index) + 1, dtype=np.object)
    for ind, val in zip(series.index, series.values):
        converted_values[ind] = val
    return converted_values.tolist()

def repair_path_token(data_list,repair_list):
    for tmp_list in data_list:
        for ind in range(len(tmp_list)):
            tmp_list[ind] = repair_list[int(tmp_list[ind])]
    return data_list

def merge_data(data1,data2):
    '''
    Given two lists, merge them together based on the items they have in common.
    
    :param data1: the first list of data
    :param data2: the list of words to be merged into data1
    :return: The indices of the data2 elements that are not in data1.
    '''
    tmp_list=[]
    for i in range(len(data2)):
        if (data2[i] in data1):
            ind = data1.index(data2[i])
            tmp_list.append(ind)
        else:
            data1.append(data2[i])
            tmp_list.append(len(data1) - 1)
    return tmp_list,data1

def save_csv(data,csv_name):
    with open(out_path+csv_name+"s.csv","w+", encoding='utf8') as fw:
        for i in range(len(data)):
            if i==0:
                fw.writelines("id,"+csv_name+"\n")
            else:
                fw.writelines("%d,%s\n"%(i,data[i]))

#============token===============
#1. Update ID of tokens in tokens.csv
tokens_repair_list,_tokens1 = merge_data(_load_tokens(tokens_file1),_load_tokens(tokens_file2))
print("tokens_repair_list: ",tokens_repair_list)
save_csv(_tokens1,"token")

#============node_types============
#2. Update ID of node_types in node_types.csv
node_types_repair_list,_node_types1 = merge_data(_load_node_types(node_types_file1),_load_node_types(node_types_file2))
print(node_types_repair_list)
save_csv(_node_types1,"node_type")


#============paths==============
#3. Update ID of paths in paths.csv
_paths1 = _load_paths(paths_file1)
_paths2 = _load_paths(paths_file2)

for i in range(1,len(_paths2)):
    tmp_list = _paths2[i].split(' ')
    for ind in range(len(tmp_list)):
        tmp_list[ind] = node_types_repair_list[int(tmp_list[ind])]
    _paths2[i] = "".join(map(lambda x:str(x)+" ",tmp_list))[:-1]

paths_repair_list,_paths1 = merge_data(_paths1,_paths2)
print("paths_repair_list: ",paths_repair_list)
save_csv(_paths1,"path")

#=================path_tokens===========
#4. Update path and token in path_tokens.csv
_original_labels, _tokens_by_author, _paths_by_author = _load_rf_contexts_file(path_tokens_file2)
_tokens_by_author = repair_path_token(_tokens_by_author,tokens_repair_list)
_paths_by_author = repair_path_token(_paths_by_author,paths_repair_list)
with open(out_path+"path_tokens.csv","w+", encoding='utf8') as fw:
    for label_index in range(len(_original_labels)):
        fw.write(_original_labels[label_index])
        tmp_token_path_str=""
        for token in _tokens_by_author[label_index]:
            tmp_token_path_str += " token,"+str(token)
        for path in _paths_by_author[label_index]:
            tmp_token_path_str += " path,"+str(path)
        fw.write(tmp_token_path_str)
        if label_index != len(_original_labels) - 1:
            fw.write("\n")

#==============path_contexts==============================
#5. Update start,path and end in path_contexts.csv
labels, start_tokens,path,end_tokens = _load_path_contexts_files(path_contexts_file2)
start_tokens = repair_path_token(start_tokens,tokens_repair_list)
path = repair_path_token(path,paths_repair_list)
end_tokens = repair_path_token(end_tokens,tokens_repair_list)

with open(out_path+"path_contexts.csv","w+", encoding='utf8') as fw:
    for label_index in range(len(labels)):
        fw.write(labels[label_index])
        for ind in range(len(start_tokens[label_index])):
            fw.write(" %d,%d,%d"%(start_tokens[label_index][ind], path[label_index][ind] ,end_tokens[label_index][ind]))
        fw.write("\n")
