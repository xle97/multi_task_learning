"""
该脚本用于测试精度， 预测以及导出onnx
"""
import os
import glob
import torch
import argparse
from model import MultiTaskModel
from torchvision import transforms

index2name0 = {0:"neutral", 1:"sexy", 2:"porn", 3:"hentai", 4:"drawings"}
index2name1 = {0:"neutral", 1:"youth_emblem", 2:"cpc_emblem", 
            3:"other_national_flag", 4:"national_emblem",
            5:"meme", 6:"national_flag"}
index2name2 = {0:"neutral", 1:"games", 2:"bloody", 3:"riot", 4:"corpse",
            5:"terrorist", 6:"burning_explosion", 7:"crowd",
            8:"armed_non_terrorist"}

def parser():
    parser = argparse.ArgumentParser(description="Multi task for pretrained model")
    parser.add_argument('--task', choices=["predict", "eval", "export"],
                    help='specify the type of task to be executed')
    parser.add_argument('--ckpt', default='resnet34',
                        help='checkpoint to be use')
    parser.add_argument('--arch', default='resnet34',
                        help='model arch')                   
    parser.add_argument('--gpu', default=-1, type=int,
                        help='-1: use cpu, others: gpu id to use')
    parser.add_argument('--work-dir', default="work_dir", 
                        help='path to save file')
    parser.add_argument('--data', default="test_data", 
                        help='data path for prediction or evaluation')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    args = parser.parse_args()
    return args

def validate(val_loader, model, args):
    from main import accuracy
    def run_validate(loader):
        with torch.no_grad():
            for i, (images, target) in enumerate(loader):
                if args.gpu is not None and torch.cuda.is_available():
                    images = images.cuda(args.gpu, non_blocking=True)
                if torch.cuda.is_available():
                    sexy = target[0].cuda(args.gpu, non_blocking=True)
                    flag = target[1].cuda(args.gpu, non_blocking=True)
                    violence = target[2].cuda(args.gpu, non_blocking=True)
                    target = (sexy, flag, violence)

                # compute output
                output = model(images)

                acc_1, acc_2, acc_3 = accuracy(output, target)
                # acc_sexy.append(acc_1.item(), images.size(0))
                # acc_flag.append(acc_2.item(), images.size(0))
                # acc_violence.append(acc_3.item(), images.size(0))
                acc_sexy.append(acc_1.item())
                acc_flag.append(acc_2.item())
                acc_violence.append(acc_3.item())

    acc_sexy = []
    acc_flag = []
    acc_violence = []

    run_validate(val_loader)
    print(sum(acc_sexy)/len(acc_sexy))
    print(sum(acc_flag)/len(acc_flag))
    print(sum(acc_violence)/len(acc_violence))


def main(args):
    if args.gpu == -1:
        device = torch.device("cpu")
    else:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.gpu)
            device = torch.device("cuda")
        else:
            raise Exception("no cuda can be used")
    model = MultiTaskModel(args.arch)
    checkpoint = torch.load(args.ckpt, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if args.task == "export":
        img = torch.randn(1, 3, 224, 224)
        save_path = args.work_dir+"/mtl.onnx"
        torch.onnx.export(model, img, save_path, input_names=["input"], opset_version=11)
        print("export done")
        return
    if args.task == "predict":
        from PIL import Image
        model = model.to(device)
        data_list = glob.glob(args.data+"/*")
        tfms = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        for data in data_list:
            img = Image.open(data).convert('RGB')
            img = tfms(img)
            img = img.unsqueeze(0)  # 增加batch维度
            img = img.to(device)
            logits = model(img)

            print(data)
            _, pred = torch.max(logits[0].data,1)
            print("涉黄分类:", index2name0[pred.item()])
            _, pred = torch.max(logits[1].data,1)
            print("涉政分类:", index2name1[pred.item()])
            _, pred = torch.max(logits[2].data,1)
            print("涉恐分类:", index2name2[pred.item()])
            
    else:
        model = model.to(device)
        val_pathlist = []
        val_labellist = []
        with open(os.path.join(args.data, 'val.txt'), 'r') as fr:
            for line in fr:
                path, l1, l2, l3 =line.strip().split()
                val_pathlist.append(path)
                val_labellist.append([l1, l2, l3])
        from dataset import MultiTaskDataset
        val_dataset = MultiTaskDataset(val_pathlist, val_labellist)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, sampler=None)

        validate(val_loader, model, args)

if __name__ == "__main__":
    args =  parser()
    main(args)