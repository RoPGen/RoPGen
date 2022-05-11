import sys
import os

# FIXME #1: Change this to your project root.
repo_path = '/home/hx/data/code-imitator/'
sys.path.insert(1, os.path.join(repo_path, 'src', 'PyProject'))

from featureextractionV2.StyloFeaturesProxy import StyloFeaturesProxy
from featureextractionV2.StyloFeatures import StyloFeatures
from featureextractionV2.StyloUnigramFeatures import StyloUnigramFeatures

import numpy as np
import pickle

from ConfigurationLearning.ConfigurationLearningRNN import ConfigurationLearningRNN
import ConfigurationGlobalLearning as Config
from classification.NovelAPI.Learning import Learning

import tensorflow as tf
from keras import backend as K
from tensorflow.keras.models import load_model


# config = tf.ConfigProto(device_count = {'GPU': 0})
# session = tf.Session(config=config)
# K.set_session(session)

os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=config)

np.set_printoptions(threshold=sys.maxsize)
############ Input
# parser = argparse.ArgumentParser(description='Start Attack')
# parser.add_argument('problemid', type=str, nargs=1,
#                    help='the problem id')
# parser.add_argument('gpu', type=str, nargs=1,
#                    help='the gpu to be used')
# args = parser.parse_args()
# PROBLEM_ID_LOADED = args.problemid[0]
# PROBLEM_ID_LOADED = "3264486_5736519012712448"
# GPU_LOADED = args.gpu[0]
# GPU_LOADED = "0"
# print("Loaded:", PROBLEM_ID_LOADED, " with GPU:", GPU_LOADED)


# Further parameters:
# we use the following dataset.
#datasetpath = os.path.join(repo_path, "data", "train")
#testpath = os.path.join(repo_path, "data", "testset")

# FIXME #2: Change this to your testing set.
testattackpath = os.path.join(repo_path, "data", "testattack_cppgcj_folds", "fold1")
#testattackpath = os.path.join(repo_path, "data", "dataset_2017", "dataset_test_t_githubc", "fold1")
#testattackpath = '/home/hx/data/Mis/media/targeted_3264486_5736519012712448/mcts'
#testattackpath = '/home/hx/data/code-imitator/data/testattack_githubc_folds/fold1'
#testattackpath = '/home/hx/data/code-imitator/data/testattack_gcjjava/fold1'

# we specify some stop words, see ConfigurationGlobalLearning.py
stop_words_codestylo = ["txt", "in", "out", "attempt0", "attempt", "attempt1", "small", "output", "input"]
# We assume 8 files per author
probsperprogrammer = 8
# we assume a dataset of 204 authors in total
no_programmers = 204

############### Variable Definition ##############

config_learning: ConfigurationLearningRNN = ConfigurationLearningRNN(
    repo_path=Config.repo_path,
    dataset_features_dir=os.path.join(Config.repo_path, "data/dataset_2017"),
    suffix_data="_2017_8_formatted_macrosremoved",
    learnmodelspath=Config.learnmodelspath,
    use_lexems=False,
    use_lexical_features=False,
    stop_words=Config.stop_words_codestylo,
    probsperprogrammer=Config.probsperprogrammer,
    no_of_programmers = 204,
    noofparallelthreads=8,
    scale=True,
    cv_optimize_rlf_params=False,
    cv_use_rnn_output=False,
    hyperparameters={
                      "RNN_epochs": [100, 200, 300, 350, 450, 500], #350], #50],
                      "RNN_nounits": [32, 128, 196, 256, 288], #, feature_dim],
                      "RNN_dropout": [0.6],
                      "RNN_lstmlayersno": [3],
                      "RNN_denselayersno": [3],
                      "RNN_l2reg": [0.00001],
                      "RNN_denseneurons": [0.45]
                      }
)


threshold_sel: int = 800
learn_method: str = "RNN"

learning: Learning = Learning()
threshold = 800

model = pickle.load(open(sys.argv[1], 'rb'))
keras_model = load_model(sys.argv[2])
model.add_rnn(keras_model)
# FIXME #3: Change this if you want to use RNN output instead of RF output.
model.use_rlf = True
trainfiles = model.data_final_train

# FIXME #4: Add your own list of authors if you're training on your own dataset
# (see the "Testing" section in `<project root>/gradaug_readme.md` for details).
if len(sys.argv) <= 3 or sys.argv[3] == 'cpp':
  authors = ['0x03BB', '4yn', 'ACMonster', 'ALOHA.Brcps', 'Ali.Sh', 'Alireza.bh', 'Alpgc',
'AnonymousBunny', 'Astein', 'Balajiganapathi', 'Birukhatri', 'DAle',
'David.Liu', 'Efgen', 'EiLx2', 'EricXu', 'FedePousa', 'Grandrogue', 'Gromah',
'GuaiNiGuoFenMeiLi', 'HTC', 'House93', 'Hurski', 'ImBarD', 'JAYS', 'Jael860',
'KhaleD', 'L3Sota', 'LEcry', 'LilacLu', 'Loud.Scream', 'LybF', 'Manro',
'MaxKalininMS', 'Melanie.Dong', 'MikeZZZ', 'MiriTheRing', 'Miseri',
'Mucosolvan', 'Mysteryname', 'OKuang', 'Phantoms', 'Plypy', 'Qumeric',
'Raycosine', 'RockyB', 'Scomip', 'ShayanH', 'Shloub', 'Shuto', 'SiruPorong',
'SlavaSSU', 'StillFantasy', 'SummerDAway', 'Tashi711', 'Tashiqi', 'Tblock',
'Thanabhat', 'TheEnglishMajor27', 'TrePe', 'TrueBamboo', 'TuanNM', 'TuneDere',
'TungNP', 'UncleGrandpa', 'VGel', 'WCG', 'Wellan', 'Witalia', 'YJWD', 'Yoshine',
'ZanderShah', 'ZooL', 'abhisheksaini', 'abisheka', 'aki33524', 'akulsareen',
'alin42', 'aman.chandna', 'angwuy', 'bennikartefla', 'bigelephant29',
'bogeyman', 'burden', 'cabinfever', 'ccsnoopy', 'chaplin', 'chocimir',
'codemonger', 'csegura', 'daninnm', 'dariofg', 'davi0015', 'disneyp',
'dothanhlam97', 'eagle93', 'elin42', 'emofunc', 'eugenus', 'evenharder',
'evgenstf', 'femto', 'fidels', 'fragusbot', 'frankbozar', 'fswenton',
'georgevidalakis', 'gultai4ukr', 'hayassy', 'hhhhhhhhhhhhhhhhhhhhhh',
'hikarico', 'hmich', 'hoangtuanh180593', 'iPeter', 'ibrahim5253', 'ihahi',
'imulan', 'ion', 'iwashi31', 'jasonwang924', 'jddantes', 'jiang.zhi', 'jiian',
'jodik', 'killjee', 'king1224', 'kjp4155', 'kojingharang', 'kuzphi', 'kyleyip',
'kzoacn', 'ladpro98', 'lan496', 'lastonephy', 'lazyBit', 'likecs', 'liymouse',
'liyufeng', 'luki4824', 'luucasv', 'manjunath1996', 'marcospqf', 'mengrao',
'mickeyandkaka', 'minaminao', 'moonbing', 'nein', 'nofto', 'nonsequitur',
'ojasdeshpande', 'okaduki', 'petwill', 'pipishrimp0505', 'pkwv', 'poao',
'purupuyo', 'qingl', 'rodz', 'sam721', 'sammyMaX', 'satos', 'satyaki3794',
'sazerterus', 'sdya', 'siqisiqi', 'skavurskaa', 'sohelH', 'spencer', 'sping128',
'splucs', 'squeespoon', 'ss1h2a3tw', 'ssor96', 'st.ni', 'stoness',
'tangziyi001', 'tdang33', 'teddytao18', 'thatprogrammer', 'tony810430', 'try',
'u8765', 'valor11', 'vector310', 'verngutz', 'vexorian', 'vladislavbelov',
'vntshh', 'vsp4', 'vstrimaitis', 'vudduu', 'watlac', 'wmpeng',
'woodpecker112358', 'wwt15', 'xyiyy', 'yeongjinc', 'ylc1218', 'yosss',
'yunmagz', 'zapray', 'zhiheng3', 'zizhong', 'zjyhala']
elif sys.argv[3] == 'gcj':
    authors = ['AhmadMamdouh', 'AhmedFathyAly', 'Ajlohr', 'Arup', 'BlueBear', 
'Bradrin', 'ChrisA', 'EgorKulikov', 'Elias', 'Gleb', 'Kirhog', 'Kristofer', 
'Lewin', 'Mingbo', 'OMGTallMonster', 'Ratmir15', 'RogerB', 'Samjay', 'Sasha', 
'TrungHieu11', 'VArtem', 'Wolfje', 'Yarin', 'andreyd', 'antonkovsharov', 'billtoth', 
'bohuss', 'ctunoku', 'cyon', 'dalex', 'darnley', 'eatmore', 'fanKarpaty', 'ferryabt',
'hamadu', 'heekyu', 'hiro116s', 'hs484', 'it3', 'ivanpopelyshev', 'jchen314', 
'jeffreyxiao', 'jerdno', 'johnnyhibiki', 'kubusgol', 'kyc', 'lucasr', 'migueldurazo', 
'mikigergely', 'monyone', 'nickbuelich', 'palys', 'pashka', 'paulliu', 'piroz', 
'qwerty787788', 'rabot', 'rogerfgm', 'rzheng', 'slex', 'stolis', 'tafit3', 'tanzaku', 
'travm12', 'trold', 'trungpham90', 'tsukuno', 'uwi', 'victorxu', 'vincentbelrose', 
'xiaowuc1', 'ykabaran', 'yo35', 'zaphod']
elif sys.argv[3] == 'java40':
  authors = ['0opslab', 'Asrar_Ahmed_Makrani', 'BrandConstantin', 'Josh_Code', 'Max_Lynch',
'Muhammad_Wasif_Javed', 'Tobias_Ogallo', 'Tran_Dinh_Hop', 'Viscent',
'andengineexamples', 'applewjg', 'bethrobson', 'brianway', 'chacha',
'chao420456', 'chweixin', 'dlna_framework', 'emigonza', 'federicodotta',
'golangpkg', 'hdzg', 'java_project_jar', 'jerry_m_lumontod', 'jfpl',
'johnno1962', 'lemire', 'mark_watson', 'mthli', 'newweb', 'pacman', 'pengrad',
'quyi', 'sample_server', 'seadroid', 'terma', 'vlc_android_macbuild',
'wak_edil', 'waimai', 'weather', 'weijuhui', '0opslab']
elif sys.argv[3] == 'java40_20':
  authors = ['0opslab', 'Viscent', 'andengineexamples', 'chacha', 'chao420456', 'chweixin',
  'dlna_framework', 'golangpkg', 'hdzg', 'jfpl', 'lemire', 'mthli', 'newweb', 'pacman', 'quyi',
  'sample_server', 'seadroid', 'terma', 'waimai', 'weijuhui']
elif sys.argv[3] == 'githubc36':
  authors = ['0712023', '254Odeke', 'Dhruvik-Chevli', 'GirijalaAditya', 'JeyaramanOfficial', 'MFarid94',
  'MuhammadAlaminMir', 'Oryx-Embedded', 'Qu-Xiangjun', 'RobertoBenjami', 'RobsonRafaeldeOliveiraBasseto',
  'SugumaranEvil', 'TSN-SHINGENN', 'ael-bagh', 'apoorvasrivastava98', 'ashlyn2002', 'behergue', 'bgmanuel99',
  'christiane-millan', 'dishanp', 'earth429', 'ezidol', 'fikepaci', 'flora0110', 'fotahub', 'ianliu98',
  'kalpa96', 'mehedi9021', 'paawankohli', 'revathy16296', 'rgautam320', 'saturneric', 'sdukesameer',
  'shruti-sureshan', 'theuwis', 'zjzj-zz']
elif sys.argv[3] == 'githubc67':
  authors = ['0712023', '254Odeke', '2security', '4rslanismet', 'Ana-Morales', 'Cz8rT',
  'DanielSalis', 'Dhruvik-Chevli', 'DiegoMendezMedina', 'GirijalaAditya', 'HakNinja',
  'JeyaramanOfficial', 'MFarid94', 'MartinMarinovich', 'Mr-JoE1', 'Oryx-Embedded', 'Qu-Xiangjun',
  'RafaelFelisbino-hub', 'RaigoXD', 'RobertoBenjami', 'RobsonRafaeldeOliveiraBasseto',
  'Sowmyamithra', 'SugumaranEvil', 'TSN-SHINGENN', 'Theemiss', 'abhijeetmurmu1997', 'ael-bagh', 'andi-s0106',
  'ankitraj311', 'apoorvasrivastava98', 'ashlyn2002', 'augustogunsch', 'behergue', 'bgmanuel99',
  'chandanXP', 'christiane-millan', 'davibernardos', 'deepaliajabsingjadhav', 'deessee0',
  'dishanp', 'dle2005', 'earth429', 'fikepaci', 'flora0110', 'fotahub', 'gokulsreekumar',
  'haon1026', 'henrique-tavares', 'jdes01', 'jimmywong2003', 'jose120918', 'kalpa96', 'kbtomic',
  'mandarvu', 'mehedi9021', 'paawankohli', 'qtgeo1248', 'revathy16296', 'rgautam320', 'ria3999',
  'sahadipanjan6', 'sdukesameer', 'seefeesaw', 'shengelenge', 'tadeograch', 'theuwis', 'zjzj-zz']
elif sys.argv[3] == 'gcj39':
  authors = ['AhmadMamdouh', 'AhmedFathyAly', 'Ajlohr', 'Arup', 'BlueBear', 'Bradrin', 'ChrisA', 'EgorKulikov', 'Elias', 'Gleb', 'Ratmir15', 'Yarin', 'andreyd', 'antonkovsharov', 'billtoth', 'bohuss', 'ctunoku', 'cyon', 'dalex', 'darnley', 'eatmore', 'fanKarpaty', 'ferryabt', 'hamadu', 'heekyu', 'hiro116s', 'hs484', 'it3', 'ivanpopelyshev', 'jchen314', 'migueldurazo', 'mikigergely', 'nickbuelich', 'piroz', 'stolis', 'trold', 'tsukuno', 'victorxu', 'vincentbelrose', 'yo35']
elif sys.argv[3] == 'cpp20':
  authors = ['4yn', 'ACMonster', 'ALOHA.Brcps', 'Alireza.bh', 'DAle', 'ShayanH', 'SummerDAway',
  'TungNP', 'aman.chandna', 'ccsnoopy', 'chocimir', 'csegura', 'eugenus', 'fragusbot', 'iPeter',
  'jiian', 'liymouse', 'sdya', 'thatprogrammer', 'vudduu']
testset_list = []
if len(sys.argv) > 4 and sys.argv[4] == '-g':
  gen_testset_list = True
else:
  gen_testset_list = False
  if len(sys.argv) > 4:
    if sys.argv[4] == '-f':
      list_file = os.path.join(testattackpath, 'test.txt')
    else:
      list_file = sys.argv[4]
    for line in open(list_file, 'r'):
      testset_list.append(line)

unigrammmatrix_test = StyloUnigramFeatures(inputdata=testattackpath,
                                        nocodesperprogrammer=1,
                                        noprogrammers=20, # FIXME #5: Change this to the number of authors in the testing set.
                                        binary=False, tf=True, idf=True,
                                        ngram_range=(1, 3), stop_words=stop_words_codestylo,
                                        trainobject=trainfiles.codestyloreference)
testfiles: StyloFeatures = StyloFeaturesProxy(codestyloreference=unigrammmatrix_test)
testfiles.createtfidffeatures(trainobject=trainfiles)
testfiles.selectcolumns(index=None, trainobject=trainfiles)
#listoftraintestsplits = learning.do_local_train_test_split(train_obj=trainfiles, config_learning=config_learning,
#                                                             threshold=threshold, trainproblemlength=None)

print(">Whole train set: TRAIN:", trainfiles.getfeaturematrix().shape[0], "TEST:", testfiles.getfeaturematrix().shape[0])
#train_obj, test_obj = learning._tfidf_feature_selection(train_obj=trainfiles, test_obj=testfiles,
#                                                    config_learning=config_learning,
#                                               threshold=threshold)

def find_author(author):
    i = 0
    for item in authors:
        if item.replace('_','') == author: return i
        if item == author or item.replace('.','') == author: return i
        i+=1
x_test = testfiles.getfeaturematrix()
y_test = testfiles.getlabels()
doclabels = testfiles.getdoclabels()
print(len(doclabels))
y_matrix = np.zeros(config_learning.no_of_programmers)
tp1 = 0
tp2 = 0
total = 0
uni_success_list = []
testset_success_list = []
for i in range(testfiles.getfeaturematrix().shape[0]):
    predicted = model.predict(x_test[i])
    
    # FIXME #6: Change how this author name extraction is done to suit the dataset you're testing on.
    #src_author = doclabels[i].split('.')[0].split('##')[-1].split('***')[0]
    #dst_author = doclabels[i].split('.')[0].split('##')[-1].split('***')[1]

    src_author = doclabels[i].split('.')[0].split('##')[-2]
    dst_author = doclabels[i].split('.')[0].split('###')[-1]

    #src_author = doclabels[i].split('_')[-2]
    #dst_author = doclabels[i].split('.cpp')[0].split('_')[-1]
    if find_author(src_author) == predicted:
        tp1+=1
    if find_author(dst_author) == predicted:
        if not gen_testset_list and testset_list != []:
            for item in testset_list:
                basename = '.'.join(item.split('##')[-1].split('.')[:-1])
                if doclabels[i].startswith(basename + '##') or doclabels[i].startswith(basename + '_'):
                    print(basename)
                    tp2+=1
                    total+=1
                    testset_success_list.append(doclabels[i])
                    break
        else:
          tp2+=1
          total+=1
          testset_success_list.append(doclabels[i])
        # prog_name = doclabels[i].split('##')[0]
        uni_success_list.append(doclabels[i])
    else:
        if not gen_testset_list and testset_list != []:
            for item in testset_list:
                basename = '.'.join(item.split('##')[-1].split('.')[:-1])
                if doclabels[i].startswith(basename + '##') or doclabels[i].startswith(basename + '_'):
                  total+=1
        else:
          total+=1
    #print(find_author(src_author), find_author(dst_author))
    print(i, 'Program: ', doclabels[i], 'Predicted class: ', predicted)
    print(i, 'Source/target author: ', src_author, dst_author, 'Predicted author: ', authors[predicted])
    # print(model.predict_proba(x_test[i], find_author(src_author))[find_author(src_author)], model.predict_proba(x_test[i], predicted)[predicted])
    # print(model.predict_proba(x_test[i], find_author(src_author))[find_author(src_author)], model.predict_proba(x_test[i], find_author(dst_author))[find_author(dst_author)], model.predict_proba(x_test[i], predicted)[predicted])
    #for j in range(config_learning.no_of_programmers):
    #    y_matrix[j] = model.predict_proba(x_test[i], j)
    #print(y_matrix)
# tp2=0
print(tp1, tp2)
print(total)
print('Targeted attack success rate='+str(tp2 / total))
if gen_testset_list:
  with open(os.path.join(testattackpath, 'test.txt'), 'w') as f:
    for item in testset_success_list:
      f.write(item + '\n')
with open('test.txt', 'w') as f:
  for item in uni_success_list:
    f.write(item + '\n')
    #print(item)
