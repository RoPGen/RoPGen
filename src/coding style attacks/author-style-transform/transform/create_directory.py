import os

auth_style = './author_style/'
xml_file = './xml_file/'
test = './program_file/test/'
target_author_file = './program_file/target_author_file/'
targeted_attack_file = './program_file/targeted_attack_file/'
untargeted_attack_file = './program_file/untargeted_attack_file/'
style = './style/transform/'

path_list = []
path_list.append(auth_style)
path_list.append(xml_file)
path_list.append(test)
path_list.append(target_author_file)
path_list.append(targeted_attack_file)
path_list.append(untargeted_attack_file)
path_list.append(style)

for path in path_list:
    if not os.path.exists(path):
        os.makedirs(path)
