from transform.py import ternary, switch_if
def get_program_style(xml_path, file_type):
    num_21_1 = switch_if.get_number(xml_path)
    num_21_2 = ternary.get_number(xml_path)
    print('21.1:', num_21_1, '21.2:', num_21_2)
    return {'21.1': num_21_1, '21.2': num_21_2}

def check_transform(auth_style, program_style, path_program, path_author, converted_styles):
    if auth_style == '21.1' and program_style['21.2'] > 0:
        converted_styles.append('21')
        ternary.program_transform(path_program)
    elif auth_style == '21.2' and program_style['21.1'] > 0:
        converted_styles.append('21')
        switch_if.program_transform(path_program)
