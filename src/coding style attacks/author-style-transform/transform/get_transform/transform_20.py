from transform.py import for_while, while_for

def get_possible_styles():
	return [1,2]

def get_instances_and_styles(e, root):
	instances_and_styles = []
	fors = for_while.get_for(e)
	whiles = while_for.get_while(e)
	for instance in fors:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 18, 1))
	for instance in whiles:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 18, 2))
	return instances_and_styles

def get_program_style(xml_path, file_type):
    num_20_1 = for_while.get_number(xml_path)
    num_20_2 = while_for.get_number(xml_path)
    print('20.1:', num_20_1, '20.2:', num_20_2)
    return {'20.1': num_20_1, '20.2': num_20_2}

def check_transform(auth_style, program_style, path_program, path_author, converted_styles):
	if auth_style == '20.1' and program_style['20.2'] > 0:
		converted_styles.append('20')
		while_for.program_transform(path_program)
	elif auth_style == '20.2' and program_style['20.1'] > 0:
		converted_styles.append('20')
		for_while.program_transform(path_program)

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