import os
from test_transform.py import typedef, retypedef
def get_program_style(file_path):
    if(typedef.get_style(file_path)):
        print('10.1: 1', '10.2: 0')
        return {"10.1": 1,"10.2": 0}
    else:
        print('10.1: 0', '10.2: 1')
        return {"10.1": 0, "10.2": 1}

def transform(prog_fullpath, target_author, orig_prog_path, save_to, ignore_list):
    import random
    per_tf_ignore_list = []
    authors_root = os.path.dirname(target_author)
    randnum = random.randint(0, len(os.listdir(authors_root)))
    if randnum == 0:
        #10.1转10.2
        e = typedef.init_parse(prog_fullpath)
        flag, doc, new_ignore_list = typedef.trans_define(e, ignore_list)
        if flag:
            per_tf_ignore_list = new_ignore_list
            typedef.save_file(doc, save_to)
    else:
        #10.2转10.1
        flag, doc, new_ignore_list = retypedef.program_transform(prog_fullpath, target_author, ignore_list)
        if flag:
            per_tf_ignore_list = new_ignore_list
            retypedef.save_file(doc, save_to)
    return per_tf_ignore_list

if __name__=='__main__':
    get_program_style('../xml_file/gerben/flyswatter.xml')