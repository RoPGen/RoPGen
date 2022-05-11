from transform.py import if_spilt, if_combine
def get_program_style(xml_path, file_type):
    num_22_1 = if_spilt.get_number(xml_path)
    num_22_2 = if_combine.get_number(xml_path)
    print('22.1:', num_22_1, '22.2:', num_22_2)
    return {'22.1': num_22_1, '22.2': num_22_2}

def check_transform(auth_style, program_style, path_program, path_author, converted_styles):
    if auth_style == '22.1' and program_style['22.2'] > 0:
        converted_styles.append('22')
        if_combine.program_transform(path_program)
    elif auth_style == '22.2' and program_style['22.1'] > 0:
        converted_styles.append('22')
        if_spilt.program_transform(path_program)