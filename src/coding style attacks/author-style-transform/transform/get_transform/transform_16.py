from transform.py import include
def get_program_style(program_path, file_type='c'):
	if file_type == 'java':
		print('16.1: 0', '16.2: 0')
		return {'16.1': 0, '16.2': 0}
	else:
		print('16.1: 1', '16.2: 0')
		return {'16.1': 1, '16.2': 0}

def check_transform(auth_style, program_style, path_program, path_author, converted_styles):
	flag_ios = include.program_transform_ios(path_program, path_author)
	if flag_ios == 1:
		converted_styles.append('16')