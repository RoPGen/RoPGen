from test_transform.py import dyn_static_mem
from test_transform.py import static_dyn_mem

def get_possible_styles():
	return [1,2]

def get_instances_and_styles(e, root):
	instances_and_styles = []
	mallocs = dyn_static_mem.get_malloc_in_decls(e)
	arrays = static_dyn_mem.get_static_mem_allocs(e)
	for instance in arrays:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 14, 1))
	for instance in mallocs:
		path_list = []
		path_list.append(root.getpath(instance))
		instances_and_styles.append((path_list, 14, 2))
	return instances_and_styles

def get_program_style(xml_path):
	e1 = static_dyn_mem.init_parser(xml_path)
	e2 = dyn_static_mem.init_parser(xml_path)
	arrays = static_dyn_mem.get_static_mem_allocs(e1)
	mallocs = dyn_static_mem.get_malloc_in_decls(e2)
	mallocs_len = len(mallocs)
	arrays_len = len(arrays)
	print('14.1:', arrays_len, '14.2:', mallocs_len)
	return {'14.1': arrays_len, '14.2': mallocs_len}

def transform(e, path_list, src_style, dst_style, save_to, ignore_list):
	#print(path_list)
	instances = [[e(path)[0] for path in path_list if len(e(path)) > 0]]
	#print(instances)
	per_tf_ignore_list = []
	if dst_style == 1:
		flag, doc, new_ignore_list = dyn_static_mem.dyn_to_static(e, ignore_list, instances)
		if flag:
			per_tf_ignore_list = new_ignore_list
			dyn_static_mem.save_tree_to_file(doc, save_to)
	elif dst_style == 2:
		flag, doc, new_ignore_list = static_dyn_mem.static_to_dyn(e, ignore_list, instances)
		if flag:
			per_tf_ignore_list = new_ignore_list
			static_dyn_mem.save_tree_to_file(doc, save_to)
	return per_tf_ignore_list