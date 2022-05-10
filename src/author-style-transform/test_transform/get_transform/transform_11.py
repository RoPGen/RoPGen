import os
from test_transform.py import type_define, type11_def

def get_program_style(file_path):
    if(type_define.get_style(file_path)):
        print('11.1: 1', '11.2: 0')
        return {"11.1": 1, "11.2": 0}
    else:
        print('11.1: 0', '11.2: 1')
        return {"11.1": 0, "11.2": 1}

def transform(prog_fullpath, target_author, orig_prog_path, save_to, ignore_list):
    import random
    per_tf_ignore_list = []
    authors_root = os.path.dirname(target_author)
    randnum = random.randint(0, len(os.listdir(authors_root)))
    if randnum == 0:
        #11.1转11.2
        e = type_define.init_parse(prog_fullpath)
        flag, doc, new_ignore_list = type_define.trans_define(e, ignore_list)
        if flag:
            per_tf_ignore_list = new_ignore_list
            type_define.save_file(doc, save_to)
    else:
        #11.2转11.1
        flag, doc, new_ignore_list = type11_def.program_transform(prog_fullpath, target_author, ignore_list)
        if flag:
            per_tf_ignore_list = new_ignore_list
            type11_def.save_file(doc, save_to)
    return per_tf_ignore_list
if __name__=='__main__':
    get_program_style('../xml_file/flym/za.xml')