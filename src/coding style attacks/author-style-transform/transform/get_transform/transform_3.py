from transform.py import select_nontmp_id_names

def transform(prog_fullpath, target_author, orig_prog_path, save_to, ignore_list):
	per_tf_ignore_list = []
	new_ignore_list = select_nontmp_id_names.transform_nontmp_id_names(prog_fullpath, target_author, ignore_list, save_to)
	if len(new_ignore_list) > 0:
		per_tf_ignore_list = new_ignore_list
	return per_tf_ignore_list

def	get_program_style(xml_path, file_type):
	print('3.1: 1', '3.2: 0')
	return {'3.1': 1, '3.2': 0}

def check_transform(auth_style, program_style, path_program, path_author, converted_styles):
	if program_style[auth_style] > 0:
		if select_nontmp_id_names.is_transformable(path_program, path_author):
			converted_styles.append('3')
			select_nontmp_id_names.program_transform(path_program, path_author)

