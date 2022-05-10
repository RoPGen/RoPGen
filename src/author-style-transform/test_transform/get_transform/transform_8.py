from test_transform.py import array_to_pointer, pointer_to_array

def get_possible_styles():
	return [1,2]

def get_instances_and_styles(e, root):
	instances_and_styles = []
	style1 = array_to_pointer.get_index(e)
	style2 = pointer_to_array.get_expr(e)
	for instance in style1:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 8, 1))
	for instance in style2:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 8, 2))
	return instances_and_styles

def get_program_style(xml_path):
    num1 = array_to_pointer.countnum(xml_path)
    num2 = pointer_to_array.countnum(xml_path)
    print("8.1: %d 8.2: %d" % (num1, num2))
    return {'8.1': num1, '8.2': num2}

def transform(e, path_list, src_style, dst_style, save_to, ignore_list):
	#print(path_list)
	instances = [[e(path)[0] for path in path_list if len(e(path)) > 0]]
	#print(instances)
	per_tf_ignore_list = []
	if dst_style == 1:
		flag, doc, new_ignore_list = pointer_to_array.transform(e, ignore_list, instances)
		if flag:
			per_tf_ignore_list = new_ignore_list
			pointer_to_array.save_tree_to_file(doc, save_to)
	elif dst_style == 2:
		flag, doc, new_ignore_list = array_to_pointer.transform(e, ignore_list, instances)
		if flag:
			per_tf_ignore_list = new_ignore_list
			array_to_pointer.save_tree_to_file(doc, save_to)
	return per_tf_ignore_list
