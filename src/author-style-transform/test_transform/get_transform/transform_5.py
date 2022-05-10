from test_transform.py import re_temp, temporary_var

def get_possible_styles():
	return [1,2]

def get_instances_and_styles(e, root):
	instances_and_styles = []
	re_temps = re_temp.get_instances(e)
	temporary_vars = temporary_var.get_instances(e)
	for instance_tuple in temporary_vars:
		path_list = []
		path_list.append((root.getpath(instance_tuple[0]), instance_tuple[1], instance_tuple[2], instance_tuple[3]))
		instances_and_styles.append((path_list, 5, 1))
	for instance_tuple in re_temps:
		path_list = []
		path_list.append((root.getpath(instance_tuple[0]), instance_tuple[1], instance_tuple[2], instance_tuple[3]))
		instances_and_styles.append((path_list, 5, 2))
	return instances_and_styles

def get_program_style(file_path):
	l1=temporary_var.get_style(file_path)
	l2=re_temp.get_style(file_path)
	re={l1[0]:l1[1],l2[0]:l2[1]}
	print('5.1:', l1[1], '5.2:', l2[1])
	return re

def transform(e, path_list, src_style, dst_style, save_to, ignore_list):
	#print(path_list)
	instances = [[(e(path[0])[0], path[1], path[2], path[3]) for path in path_list if len(e(path[0])) > 0]]
	#print(instances)
	per_tf_ignore_list = []
	if dst_style == 1:
		flag, doc, new_ignore_list = re_temp.trans_temp_var(e, ignore_list, instances)
		if flag:
			per_tf_ignore_list = new_ignore_list
			re_temp.save_file(doc, save_to)
	elif dst_style == 2:
		flag, doc, new_ignore_list = temporary_var.trans_temp_var(e, ignore_list, instances)
		if flag:
			per_tf_ignore_list = new_ignore_list
			temporary_var.save_file(doc, save_to)
	return per_tf_ignore_list