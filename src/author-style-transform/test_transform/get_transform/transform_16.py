from test_transform.py import for_while, while_for

def get_possible_styles():
	return [1,2]

def get_instances_and_styles(e, root):
	instances_and_styles = []
	fors = for_while.get_for(e)
	whiles = while_for.get_while(e)
	for instance in fors:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 16, 1))
	for instance in whiles:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 16, 2))
	return instances_and_styles

def get_program_style(xml_path):
    num_16_1 = for_while.get_number(xml_path)
    num_16_2 = while_for.get_number(xml_path)
    print('16.1:', num_16_1, '16.2:', num_16_2)
    return {'16.1': num_16_1, '16.2': num_16_2}

def transform(e, path_list, src_style, dst_style, save_to, ignore_list):
	#print(path_list)
	instances = [[e(path)[0] for path in path_list if len(e(path)) > 0]]
	#print(instances)
	per_tf_ignore_list = []
	if dst_style == 1:
		flag, doc, new_ignore_list = while_for.trans_tree(e, ignore_list, instances)
		if flag:
			per_tf_ignore_list = new_ignore_list
			while_for.save_tree_to_file(doc, save_to)
	elif dst_style == 2:
		flag, doc, new_ignore_list = for_while.trans_tree(e, ignore_list, instances)
		if flag:
			per_tf_ignore_list = new_ignore_list
			for_while.save_tree_to_file(doc, save_to)
	return per_tf_ignore_list