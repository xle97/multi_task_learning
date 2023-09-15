import os
import glob

def step1(target_path):
    """
    查看数据分布
    涉政 7个类别(6+1)
    涉恐 9个类别(8+1)
    涉黄 5个类别(4+1)
    youth_emblem 145
    cpc_emblem 139
    other_national_flag 261
    national_emblem 186
    meme 393
    national_flag 323
    games 586
    bloody 231
    riot 366
    corpse 356
    terrorist 167
    burning_explosion 396
    crowd 775
    armed_non_terrorist 164
    sexy 269
    neutral 776
    porn 698
    hentai 101
    drawings 378
    """
    
    fd_list = glob.glob(target_path+"/*")
    for fd in fd_list:
        subfd_list = glob.glob(fd+"/*")
        for subfd in subfd_list:
            filelist = os.listdir(subfd)
            print(os.path.basename(subfd), len(filelist))

def step2(target_path):
    """
    去除字符串的空格
    """
    fd_list = glob.glob(target_path+"/*")
    for fd in fd_list:
        subfd_list = glob.glob(fd+"/*")
        for subfd in subfd_list:
            filelist = glob.glob(subfd+"/*")
            for file in filelist:
                if ' ' in file:
                    print(file)


def step3(target_path):
    """
    制作训练集、测试集, 以及相应txt
    比例默认8比2
    三个头一起训练时， 标签为互斥的
    标签映射：
        涉黄：
            neutral:0, sexy:1, porn:2, hentai:3, drawings:4
        涉恐：
            neutral:0, games:1, bloody:2, riot:3, corpse:4,
            terrorist:5, burning_explosion:6, crowd:7,
            armed_non_terrorist: 8
        涉政：
            neutral:0, youth_emblem:1, cpc_emblem:2, 
            other_national_flag:3, national_emblem:4,
            meme:5, national_flag:6
    """
    import random
    data_path_nsfw = os.path.join(target_path, "nsfw")
    data_path_politics = os.path.join(target_path, "politics")
    data_path_terrorism = os.path.join(target_path, "terrorism")
    nsfw_dict = {"neutral":"0", "sexy":"1", "porn":"2", "hentai":"3", "drawings":"4"}
    politics_dict = {"neutral":"0", "youth_emblem":"1", "cpc_emblem":"2", 
            "other_national_flag":"3", "national_emblem":"4",
            "meme":"5", "national_flag":"6"}
    terrorism_dict = {"neutral":"0", "games":"1", "bloody":"2", "riot":"3", "corpse":"4",
            "terrorist":"5", "burning_explosion":"6", "crowd":"7",
            "armed_non_terrorist": "8"}
    textL = []

    nsfw_list = os.listdir(data_path_nsfw)
    for fd in nsfw_list:
        file_list = glob.glob(os.path.join(data_path_nsfw, fd)+"/*")
        for file in file_list:
            textL.append(file+f" {nsfw_dict[fd]} 0 0")

    politics_list = os.listdir(data_path_politics)
    for fd in politics_list:
        file_list = glob.glob(os.path.join(data_path_politics, fd)+"/*")
        for file in file_list:
            textL.append(file+f" 0 {politics_dict[fd]} 0")

    terrorism_list = os.listdir(data_path_terrorism)
    for fd in terrorism_list:
        file_list = glob.glob(os.path.join(data_path_terrorism, fd)+"/*")
        for file in file_list:
            textL.append(file+f" 0 0 {terrorism_dict[fd]}")

    random.shuffle(textL)
    cnt = int(len(textL)*0.8)
    train_list = textL[0:cnt]
    val_list = textL[cnt:]
    with open(os.path.join(target_path, "train.txt"), 'w') as fw:
        for line in train_list:
            fw.write(line+"\n")
    with open(os.path.join(target_path, "val.txt"), 'w') as fw:
        for line in val_list:
            fw.write(line+"\n")
    
def step4(target_path):
    """
    制作分离训练的 train.txt 和 val.txt
    比例默认8比2
    涉政和涉恐的数据需要补充中立数据进行训练
    标签映射：
        涉黄：
            neutral:0, sexy:1, porn:2, hentai:3, drawings:4
        涉恐：
            neutral:0, games:1, bloody:2, riot:3, corpse:4,
            terrorist:5, burning_explosion:6, crowd:7,
            armed_non_terrorist: 8
        涉政：
            neutral:0, youth_emblem:1, cpc_emblem:2, 
            other_national_flag:3, national_emblem:4,
            meme:5, national_flag:6
    """
    import random
    data_path_nsfw = os.path.join(target_path, "nsfw")
    data_path_politics = os.path.join(target_path, "politics")
    data_path_terrorism = os.path.join(target_path, "terrorism")
    nsfw_dict = {"neutral":"0", "sexy":"1", "porn":"2", "hentai":"3", "drawings":"4"}
    politics_dict = {"neutral":"0", "youth_emblem":"1", "cpc_emblem":"2", 
            "other_national_flag":"3", "national_emblem":"4",
            "meme":"5", "national_flag":"6"}
    terrorism_dict = {"neutral":"0", "games":"1", "bloody":"2", "riot":"3", "corpse":"4",
            "terrorist":"5", "burning_explosion":"6", "crowd":"7",
            "armed_non_terrorist": "8"}
    
    textL = []
    nsfw_list = os.listdir(data_path_nsfw)
    for fd in nsfw_list:
        file_list = glob.glob(os.path.join(data_path_nsfw, fd)+"/*")
        for file in file_list:
            textL.append(file+f" {nsfw_dict[fd]}")
    random.shuffle(textL)
    cnt = int(len(textL)*0.8)
    train_list = textL[0:cnt]
    val_list = textL[cnt:]
    with open(os.path.join(target_path, "train_1.txt"), 'w') as fw:
        for line in train_list:
            fw.write(line+"\n")
    with open(os.path.join(target_path, "val_1.txt"), 'w') as fw:
        for line in val_list:
            fw.write(line+"\n")

    textL = []
    politics_list = os.listdir(data_path_politics)
    for fd in politics_list:
        file_list = glob.glob(os.path.join(data_path_politics, fd)+"/*")
        for file in file_list:
            textL.append(file+f" {politics_dict[fd]}")
    file_list = glob.glob(os.path.join(data_path_nsfw, "neutral")+"/*")
    for file in file_list:
        textL.append(file+f" 0")
    random.shuffle(textL)
    cnt = int(len(textL)*0.8)
    train_list = textL[0:cnt]
    val_list = textL[cnt:]
    with open(os.path.join(target_path, "train_2.txt"), 'w') as fw:
        for line in train_list:
            fw.write(line+"\n")
    with open(os.path.join(target_path, "val_2.txt"), 'w') as fw:
        for line in val_list:
            fw.write(line+"\n")

    textL = []
    terrorism_list = os.listdir(data_path_terrorism)
    for fd in terrorism_list:
        file_list = glob.glob(os.path.join(data_path_terrorism, fd)+"/*")
        for file in file_list:
            textL.append(file+f" {terrorism_dict[fd]}")
    file_list = glob.glob(os.path.join(data_path_nsfw, "neutral")+"/*")
    for file in file_list:
        textL.append(file+f" 0")
    random.shuffle(textL)
    cnt = int(len(textL)*0.8)
    train_list = textL[0:cnt]
    val_list = textL[cnt:]
    with open(os.path.join(target_path, "train_3.txt"), 'w') as fw:
        for line in train_list:
            fw.write(line+"\n")
    with open(os.path.join(target_path, "val_3.txt"), 'w') as fw:
        for line in val_list:
            fw.write(line+"\n")
    print("step4 done")

def step5(target_path):
    """
    挑选图片制作校准数据集
    """
    data_path_nsfw = os.path.join(target_path, "nsfw")
    data_path_politics = os.path.join(target_path, "politics")
    data_path_terrorism = os.path.join(target_path, "terrorism")
    
    calibL = []
    nsfw_list = os.listdir(data_path_nsfw)
    for fd in nsfw_list:
        file_list = glob.glob(os.path.join(data_path_nsfw, fd)+"/*")
        range_num = 100 if fd == "neutral" else 50
        for i in range(range_num):
            calibL.append(file_list[i])
    politics_list = os.listdir(data_path_politics)
    for fd in politics_list:
        file_list = glob.glob(os.path.join(data_path_politics, fd)+"/*")
        for i in range(50):
            calibL.append(file_list[i])
    terrorism_list = os.listdir(data_path_terrorism)
    for fd in terrorism_list:
        file_list = glob.glob(os.path.join(data_path_terrorism, fd)+"/*")
        for i in range(50):
            calibL.append(file_list[i])
    with open('calib.txt','w') as fw:
        for line in calibL:
            fw.write(line+"\n")
    print("step5 done")

def step6(tpath):
    """
    制作校准数据集
    """
    import shutil

    target_path = "calib"
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    with open('calib.txt', 'r') as fr:
        for i, line in enumerate(fr):
            name = line.strip()
            suffix = os.path.splitext(name)[-1]
            try:
                shutil.copy(name, os.path.join(target_path, f"img_{str(i)}.{suffix}"))
            except:
                print(name)
                exit(0)
    print("step6 done")

def call_fun_by_str(fun_str, args):

    eval(fun_str)(args)

if __name__ == "__main__":
    args = 5
    target_path = "/data2/rzhang/mtl_data"
    call_fun_by_str(f"step{str(args)}", args=target_path)
    
