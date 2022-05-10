from test_transform.py import split_function
from test_transform.py import java_split_function
def get_program_style(author):
	avg_func_len = split_function.count_func_avg_len_by_author(author)
	avg_nesting_level = split_function.count_avg_nesting_level_by_author(author)
	print("19:", round(avg_func_len, 1), round(avg_nesting_level, 1))
	return {'19': [round(avg_func_len, 1), round(avg_nesting_level, 1)]}
def get_program_style_java(xml_path):
	avg_func_len = java_split_function.count_func_avg_len_by_author(xml_path)
	#avg_nesting_level = java_split_function.count_avg_nesting_level_by_author(xml_path)
	print("19:", round(avg_func_len, 1), 0)
	return {'19': [round(avg_func_len, 1), 0]}