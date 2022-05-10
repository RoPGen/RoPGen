from test_transform.py import if_spilt, if_combine
def get_program_style(xml_path):
    num_18_1 = if_spilt.get_number(xml_path)
    num_18_2 = if_combine.get_number(xml_path)
    print('18.1:', num_18_1, '18.2:', num_18_2)
    return {'18.1': num_18_1, '18.2': num_18_2}
