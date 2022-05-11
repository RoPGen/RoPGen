from transform.py import include
def get_program_style(program_path, file_type='c'):
    if file_type == 'java':
        print('15.1: 0', '15.2: 0')
        return {'15.1': 0, '15.2': 0}
    else:
        print('15.1: 1', '15.2: 0')
        return {'15.1': 1, '15.2': 0}

def check_transform(auth_style, program_style, path_program, path_author, converted_styles):
    flag_namespace = include.program_transform_namespace(path_program, path_author)
    if flag_namespace == 1:
        converted_styles.append('15')
