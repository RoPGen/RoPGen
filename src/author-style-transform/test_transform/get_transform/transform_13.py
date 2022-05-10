from test_transform.py import c_lib_to_cpp, cpp_lib_to_c
num_13_2 = 0
num_13_1 = 0
def get_possible_styles():
	return [1,2]
def get_instances_and_styles(e, root):
	instances_and_styles = []
	style1 = cpp_lib_to_c.get_functions(e)
	style2 = c_lib_to_cpp.get_functions(e)
	for instance in style1:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 13, 1))
	for instance in style2:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 13, 2))
	return instances_and_styles

def get_program_style(xml_path):
	num_13_2 = c_lib_to_cpp.get_number(xml_path)
	num_13_1 = cpp_lib_to_c.get_number(xml_path)
	print('13.1:', num_13_1, '13.2:', num_13_2)
	return {'13.1': num_13_1, '13.2': num_13_2}
def get_number_13_3():
	if num_13_1 == 0 and num_13_2 == 0:
		return 1
	else:
		return 0
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