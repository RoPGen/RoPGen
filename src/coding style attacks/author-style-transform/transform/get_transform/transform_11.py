import os
from transform.py import typedef, retypedef
def get_program_style(file_path, file_type='c'):
    if file_type == 'java':
        print('11.1: 0', '11.2: 0')
        return {'11.1': 0, '11.2': 0}
    else:
        if(typedef.get_style(file_path)):
            print('11.1: 1', '11.2: 0')
            return {"11.1": 1, "11.2": 0}
        else:
            print('11.1: 0', '11.2: 1')
            return {"11.1": 0, "11.2": 1}

def check_transform(auth_style, program_style, path_program, path_author, converted_styles):
    if auth_style == '11.1' and program_style['11.1'] > 0:
        retypedef.ls.clear()
        converted_styles.append('11')
        retypedef.program_transform(path_program, path_author)

def transform(prog_fullpath, target_author, orig_prog_path, save_to, ignore_list):
    import random
    per_tf_ignore_list = []
    authors_root = os.path.dirname(target_author)
    randnum = random.randint(0, len(os.listdir(authors_root)))
    if randnum == 0:
        e = typedef.init_parse(prog_fullpath)
        flag, doc, new_ignore_list = typedef.trans_define(e, ignore_list)
        if flag:
            per_tf_ignore_list = new_ignore_list
            typedef.save_file(doc, save_to)
    else:
        flag, doc, new_ignore_list = retypedef.program_transform(prog_fullpath, target_author, ignore_list)
        if flag:
            per_tf_ignore_list = new_ignore_list
            retypedef.save_file(doc, save_to)
    return per_tf_ignore_list

if __name__=='__main__':
    get_program_style('../xml_file/gerben/flyswatter.xml')