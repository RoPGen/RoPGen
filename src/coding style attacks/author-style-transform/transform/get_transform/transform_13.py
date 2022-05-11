
from transform.py import include
from transform.py import java_import

def get_program_style(program_path, file_type):
	if file_type == 'c' or file_type == 'cpp':
		print('13.1: 1', '13.2: 0')
		return {'13.1': 1, '13.2': 0}
	elif file_type == 'java':
		print('13.1: 1', '13.2: 0')
		return {'13.1': 1, '13.2': 0}

def check_transform(auth_style, program_style, path_program, path_author, converted_styles):
	pro_list_keys, len1, c_flag, auth_list_keys, pro_list_key = include.transform_include(path_program, path_author)
	if c_flag == 1:
		converted_styles.append('13')
		include.program_transform_include(path_program, path_author)
	a_13_java, b_13_java, java_flag, auth_list_keys_java,pro_list_key_java = java_import.transform_include('./style/style.xml', path_author)
	if java_flag == 1:
		converted_styles.append('13')
		java_import.program_transform(path_program, path_author)

def transform(prog_fullpath, target_author, orig_prog_path, save_to, ignore_list):
	per_tf_ignore_list = []
	flag, doc, new_ignore_list = include.program_transform(prog_fullpath, target_author, ignore_list)
	if flag:
		per_tf_ignore_list = new_ignore_list
		include.save_tree_to_file(doc, save_to)
	return per_tf_ignore_list
