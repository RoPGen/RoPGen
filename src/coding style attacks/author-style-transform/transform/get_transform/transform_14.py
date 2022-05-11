from transform.py import include

def get_program_style(program_path, file_type='c'):
	if file_type == 'java':
		print('14.1: 0', '14.2: 0')
		return {'14.1': 0, '14.2': 0}
	else:
		print('14.1: 1', '14.2: 0')
		return {'14.1': 1, '14.2': 0}

def check_transform(auth_style, program_style, path_program, path_author, converted_styles):

	flag_return = include.program_transform_return(path_program, path_author)
	if flag_return == 1:
		converted_styles.append('14')
