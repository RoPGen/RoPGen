from transform.py import var_init_merge
from transform.py import var_init_pos

def get_possible_styles():
	return [1,2]

def get_instances_and_styles(e, root):
	instances_and_styles = []
	separate_inits = var_init_merge.get_separate_inits(e)
	merged_inits = var_init_pos.get_decl_init_stmts(e)
	for instance_tuple in merged_inits:
		path_list = []
		path_list.append((root.getpath(instance_tuple[0]), instance_tuple[1], instance_tuple[2], instance_tuple[3]))
		instances_and_styles.append((path_list, 7, 1))
	for instance_tuple in separate_inits:
		path_list = []
		path_list.append((root.getpath(instance_tuple[0]), instance_tuple[1], instance_tuple[2], instance_tuple[3]))
		instances_and_styles.append((path_list, 7, 2))
	return instances_and_styles

def get_program_style(xml_path, file_type):
	e1 = var_init_merge.init_parser(xml_path)
	e2 = var_init_pos.init_parser(xml_path)
	separate_inits = var_init_merge.get_separate_inits(e1)
	merged_inits = var_init_pos.get_decl_init_stmts(e2)
	separate_inits_len = len(separate_inits)
	merged_inits_len = len(merged_inits)
	print('7.1:', merged_inits_len, '7.2:', separate_inits_len)
	return {'7.1': merged_inits_len, '7.2': separate_inits_len}

def check_transform(auth_style, program_style, path_program, path_author, converted_styles):
	if auth_style == '7.1' and program_style['7.2'] > 0:
		converted_styles.append('7')
		var_init_merge.program_transform(path_program)
	elif auth_style == '7.2' and program_style['7.1'] > 0:
		converted_styles.append('7')
		var_init_pos.program_transform(path_program)

def transform(e, path_list, src_style, dst_style, save_to, ignore_list):
	#print(path_list)
	instances = [[(e(path[0])[0], path[1], path[2], path[3]) for path in path_list if len(e(path[0])) > 0]]
	#print(instances)
	per_tf_ignore_list = []
	if dst_style == 1:
		flag, doc, new_ignore_list = var_init_merge.transform_init(e, instances, ignore_list)
		if flag:
			per_tf_ignore_list = new_ignore_list
			var_init_merge.save_tree_to_file(doc, save_to)
	elif dst_style == 2:
		flag, doc, new_ignore_list = var_init_pos.transform_standalone_stmts(e, instances, ignore_list)
		if flag:
			per_tf_ignore_list = new_ignore_list
			var_init_pos.save_tree_to_file(doc, save_to)
	return per_tf_ignore_list