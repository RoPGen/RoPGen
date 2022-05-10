from test_transform.py import assign_combine, assign_value
def get_program_style(xml_path):
    num_15_1 = assign_value.get_number(xml_path)
    num_15_2 = assign_combine.get_number(xml_path)
    print('15.1:', num_15_1, '15.2:', num_15_2)
    return {'15.1': num_15_1, '15.2': num_15_2}
