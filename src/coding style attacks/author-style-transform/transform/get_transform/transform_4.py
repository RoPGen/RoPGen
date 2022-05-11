from transform.py import const_vars

def get_program_style(file_path, file_type):
    if(const_vars.get_style(file_path)):
        print('4.1: 1', '4.2: 0')
        return {"4.1": 1, "4.2": 0}
    else:
        print('4.1: 0', '4.2: 1')
        return {"4.1": 0, "4.2": 1}

def check_transform(auth_style, program_style, path_program, path_author, converted_styles):
    if auth_style == '4.1' and program_style['4.1'] > 0:
        converted_styles.append('4')
        const_vars.program_transform(path_program, path_author)

def transform(prog_fullpath, target_author, orig_prog_path, save_to, ignore_list):
	per_tf_ignore_list = []
	flag, doc, new_ignore_list = const_vars.program_transform(prog_fullpath, target_author, ignore_list)
	if flag:
		per_tf_ignore_list = new_ignore_list
		const_vars.save_file(doc, save_to)
	return per_tf_ignore_list

if __name__=='__main__':
    get_program_style('../xml_file/flym/za.xml')