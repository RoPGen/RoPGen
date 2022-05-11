from transform.py import include
def get_program_style(program_path, file_type='c'):
    if file_type == 'java':
        print('17.1: 0', '17.2: 0')
        return {'17.1': 0, '17.2': 0}
    else:
        print('17.1: 1', '17.2: 0')
        return {'17.1': 1, '17.2': 0}

def check_transform(auth_style, program_style, path_program, path_author, converted_styles):
    flag_freopen = include.program_transform_freopen(path_program, path_author)
    if flag_freopen == 1:
        converted_styles.append('17')