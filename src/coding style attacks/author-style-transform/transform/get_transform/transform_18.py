
import re
from transform.py import c_lib_to_cpp, cpp_lib_to_c
from transform.py.array_to_pointer import program_transform
num_18_2 = 0
num_18_1 = 0
def get_possible_styles():
	return [1, 2]
def get_instances_and_styles(e, root):
	instances_and_styles = []
	style1 = cpp_lib_to_c.get_functions(e)
	style2 = c_lib_to_cpp.get_functions(e)
	for instance in style1:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 18, 1))
	for instance in style2:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 18, 2))
	return instances_and_styles

def get_program_style(xml_path, file_type='c'):
	if file_type == 'java':
		print('18.1: 0', '18.2: 0', '18.3: 0')
		return {'18.1': 0, '18.2': 0, '18.3':0}
	else:
		num_18_2 = c_lib_to_cpp.get_number(xml_path)
		num_18_1 = cpp_lib_to_c.get_number(xml_path)
		if num_18_1 == 0 and num_18_2 == 0:
			num_18_3 = 1
			print('18.1:', num_18_1, '18.2:', num_18_2, '18.3:', num_18_3)
			return {'18.1': num_18_1, '18.2': num_18_2, '18.3': num_18_3}
		else:
			print('18.1:', num_18_1, '18.2:', num_18_2, '18.3:', 0)
			return {'18.1': num_18_1, '18.2': num_18_2, '18.3': 0}


def get_number_18_3():
	if num_18_1 == 0 and num_18_2 == 0:
		return 1
	else:
		return 0

def check_transform(auth_style, program_style, path_program, path_author, converted_styles):
	if auth_style == '18.1' and program_style['18.2'] > 0:
		converted_styles.append('18')
		c_lib_to_cpp.program_transform(path_program)
	elif auth_style == '18.2' and program_style['18.1'] > 0:
		converted_styles.append('18')
		cpp_lib_to_c.program_transform(path_program)
	elif auth_style == '18.3' and program_style['18.1'] > 0:
		converted_styles.append('18')
		cpp_lib_to_c.program_transform(path_program, '18.1', '18.3')
	elif auth_style == '18.3' and program_style['18.2'] > 0:
		converted_styles.append('18')
		cpp_lib_to_c.program_transform(path_program, '18.2', '18.3')



def transform(e, path_list, src_style, dst_style, save_to, ignore_list):
	#print(path_list)
	instances = [[e(path)[0] for path in path_list if len(e(path)) > 0]]
	#print(instances)
	per_tf_ignore_list = []
	if dst_style == 1:
		flag, doc, new_ignore_list = c_lib_to_cpp.cpp_lib_to_c(e, ignore_list, instances)
		if flag:
			per_tf_ignore_list = new_ignore_list
			c_lib_to_cpp.save_tree_to_file(doc, save_to)
	elif dst_style == 2:
		flag, doc, new_ignore_list = cpp_lib_to_c.cpp_lib_to_c(e, ignore_list, instances)
		if flag:
			per_tf_ignore_list = new_ignore_list
			cpp_lib_to_c.save_tree_to_file(doc, save_to)
	return per_tf_ignore_list
