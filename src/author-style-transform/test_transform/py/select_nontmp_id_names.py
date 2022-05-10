import sys
import os
from lxml import etree
from collections import Counter
from test_transform.py import select_tmp_id_names
from timeit import default_timer as timer

ns = {'src': 'http://www.srcML.org/srcML/src',
	'cpp': 'http://www.srcML.org/srcML/cpp',
	'pos': 'http://www.srcML.org/srcML/position'}
doc = None

def init_parser(file):
	global doc
	doc = etree.parse(file)
	return doc

def save_tree_to_file(tree, file):
	with open(file, 'w') as f:
		f.write(etree.tostring(tree).decode('utf8'))

# transform non-temporary identifier names
# basically collecting identifier names of source author and target author here
# actual transformation is done in select_tmp_id_names.py
# arguments 'src_author' path containing source author srcML XMLs
# 'dst_author' path containing target author srcML XMLs
# 'ignore_list' is pretty much legacy code and can be ignored
# 'save_to' path where resulting XML should be saved to
def transform_nontmp_id_names(src_author, dst_author, ignore_list=[], save_to='tmp.xml', keep_log=False):
	broken = False
	dst_tmp, dst_tmp_vars_type = select_tmp_id_names.get_vars_cnt_by_author(dst_author)
	src_tmp, src_tmp_vars_type = select_tmp_id_names.get_vars_cnt_by_author(src_author)
	dst_all, dst_all_vars_type = select_tmp_id_names.get_vars_cnt_by_author(dst_author, tmp_only=False, need_extra_info=True)
	src_all, src_all_vars_type = select_tmp_id_names.get_vars_cnt_by_author(src_author, tmp_only=False, need_extra_info=True)
	dst_funcs = select_tmp_id_names.get_func_name_cnt_by_author(dst_author)
	src_funcs = select_tmp_id_names.get_func_name_cnt_by_author(src_author)
	src_templates = select_tmp_id_names.get_template_names_by_author(src_author)
	dst_templates = select_tmp_id_names.get_template_names_by_author(dst_author)
	dst_all += dst_funcs
	src_all += src_funcs
	intersect = set(dst_all).intersection(set(src_all))
	#print(src_all_vars_type)
	diff = {k : dst_all[k] for k in set(dst_all) - set(dst_tmp) - intersect - set(src_templates) - set(dst_templates)}
	dst_func_vars_type = {k : {'type': 'function'} for k in set(dst_funcs)}
	dst_all_vars_type.update(dst_func_vars_type)
	dst_all_vars_type = {k : dst_all_vars_type[k] for k in set(dst_all_vars_type) - set(dst_tmp) - intersect - set(src_templates) - set(dst_templates)} # exclude identifier names that source author already uses
	src_nontmp = {k : src_all[k] for k in set(src_all) - set(src_tmp) - set(src_templates)}
	src_nontmp = sorted(src_nontmp.items(), key = lambda d: d[1], reverse=True)
	diff = sorted(diff.items(), key = lambda d: d[1], reverse=False)
	# print(src_nontmp, diff)
	file_list = os.listdir(src_author) if os.path.isdir(src_author) else [src_author]
	for src_filename in file_list:
		if not src_filename.endswith('.xml'): continue
		src_file = os.path.join(src_author if os.path.isdir(src_author) else '', src_filename)
		new_ignore_list, this_broken, doc, var_replace_log = select_tmp_id_names.replace_names(src_file, src_nontmp, src_all_vars_type, diff, dst_all_vars_type, False, save_to, ignore_list, keep_log)
		if this_broken: broken = True
	if not broken:
		save_tree_to_file(doc, save_to)
	if keep_log:
		return new_ignore_list, var_replace_log
	else:
		return new_ignore_list

def is_transformable(src_author, dst_author):
	dst_tmp, dst_tmp_vars_type = select_tmp_id_names.get_vars_cnt_by_author(dst_author)
	src_tmp, src_tmp_vars_type = select_tmp_id_names.get_vars_cnt_by_author(src_author)
	dst_all, dst_all_vars_type = select_tmp_id_names.get_vars_cnt_by_author(dst_author, tmp_only=False, need_extra_info=True)
	src_all, src_all_vars_type = select_tmp_id_names.get_vars_cnt_by_author(src_author, tmp_only=False, need_extra_info=True)
	dst_funcs = select_tmp_id_names.get_func_name_cnt_by_author(dst_author)
	src_funcs = select_tmp_id_names.get_func_name_cnt_by_author(src_author)
	dst_all += dst_funcs
	src_all += src_funcs
	diff = {k : dst_all[k] for k in set(dst_all) - set(dst_tmp) - set(src_all)}
	src_nontmp = {k : src_all[k] for k in set(src_all) - set(src_tmp)}
	diff = sorted(diff.items(), key = lambda d: d[1], reverse=False)
	if len(src_nontmp) * len(diff) > 0: return True
	return False

def select_both_id_names(src_author, dst_author):
	dst_all = get_vars_cnt_by_author(dst_author, False)
	src_all = get_vars_cnt_by_author(src_author, False)
	diff = {k : dst_all[k] for k in set(dst_all) - set(src_all)}
	src_all = sorted(src_all.items(), key = lambda d: d[1], reverse=True)
	diff = sorted(diff.items(), key = lambda d: d[1], reverse=False)
	file_list = os.listdir(src_author) if os.path.isdir(src_author) else [src_author]
	for src_filename in file_list:
		if not src_filename.endswith('.xml'): continue
		src_file = os.path.join(src_author if os.path.isdir(src_author) else '', src_filename)
		replace_names(src_file, src_all, diff, False, save_to)

if __name__ == '__main__':
	src_author = sys.argv[1]
	dst_author = sys.argv[2]
	transform_nontmp_id_names(src_author, dst_author, keep_log=True)
