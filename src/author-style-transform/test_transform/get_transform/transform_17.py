from test_transform.py import ternary, switch_if
def get_program_style(xml_path):
    num_17_1 = switch_if.get_number(xml_path)
    num_17_2 = ternary.get_number(xml_path)
    print('17.1:', num_17_1, '17.2:', num_17_2)
    return {'17.1': num_17_1, '17.2': num_17_2}