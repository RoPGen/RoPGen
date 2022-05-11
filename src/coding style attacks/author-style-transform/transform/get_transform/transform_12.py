import os
from transform.py import type_define, type11_def

def get_program_style(file_path, file_type='c'):
    if file_type == 'java':
        print('12.1: 0', '12.2: 0')
        return {'12.1': 0, '12.2': 0}
    else:
        if(type_define.get_style(file_path)):
            print('12.1: 1', '12.2: 0')
            return {"12.1": 1, "12.2": 0}
        else:
            print('12.1: 0', '12.2: 1')
            return {"12.1": 0, "12.2": 1}

def check_transform(auth_style, program_style, path_program, path_author, converted_styles):
    if auth_style == '12.1':
        converted_styles.append('12')
        type11_def.program_transform(path_program, path_author)
    elif auth_style == '12.2' and program_style['12.1'] > 0:
        converted_styles.append('12')
        type11_def.program_transform(path_program, path_author)

def transform(prog_fullpath, target_author, orig_prog_path, save_to, ignore_list):
    import random
    per_tf_ignore_list = []
    authors_root = os.path.dirname(target_author)
    randnum = random.randint(0, len(os.listdir(authors_root)))
    if randnum == 0:
        e = type_define.init_parse(prog_fullpath)
        flag, doc, new_ignore_list = type_define.trans_define(e, ignore_list)
        if flag:
            per_tf_ignore_list = new_ignore_list
            type_define.save_file(doc, save_to)
    else:
        flag, doc, new_ignore_list = type11_def.program_transform(prog_fullpath, target_author, ignore_list)
        if flag:
            per_tf_ignore_list = new_ignore_list
            type11_def.save_file(doc, save_to)
    return per_tf_ignore_list

if __name__=='__main__':
    get_program_style('../xml_file/flym/za.xml')