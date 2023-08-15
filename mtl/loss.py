from torch import nn

# class MultiTaskLossWrapper(nn.Module):
#     def __init__(self, task_num):
#         super(MultiTaskLossWrapper, self).__init__()
#         self.task_num = task_num
#         self.log_vars = nn.Parameter(torch.ones((task_num)))

#     def forward(self, preds, target):
        
#         ### 可能需要to_device
#         # nn.CrossEntropyLoss().to(device)
#         sexy, flag, violence = target
#         crossEntropy = nn.CrossEntropyLoss()
        
#         loss0 = crossEntropy(preds[0], sexy)
#         loss1 = crossEntropy(preds[1], flag)
#         loss2 = crossEntropy(preds[2], violence)

#         precision0 = 0.5 / (self.log_vars[0]**2)
#         loss0 = precision0*loss0 + torch.log(1 + self.log_vars[0]**2)

#         precision1 = 0.5 / (self.log_vars[1]**2)
#         loss1 = precision1*loss1 + torch.log(1 + self.log_vars[1]**2)

#         precision2 = 0.5 / (self.log_vars[2]**2)
#         loss2 = precision2*loss2 + torch.log(1 + self.log_vars[2]**2)

#         ### 试精度加权
        
#         return torch.mean(loss0+loss1+loss2)

## 该损失会出现负数的情况
class MultiTaskLossWrapper(nn.Module):
    def __init__(self, task_num):
        super(MultiTaskLossWrapper, self).__init__()
        self.task_num = task_num
        # self.log_vars = nn.Parameter(torch.zeros((task_num)))

    def forward(self, preds, target):
        
        ### 可能需要to_device
        # nn.CrossEntropyLoss().to(device)
        sexy, flag, violence = target
        crossEntropy = nn.CrossEntropyLoss()
        
        loss0 = crossEntropy(preds[0], sexy)
        loss1 = crossEntropy(preds[1], flag)
        loss2 = crossEntropy(preds[2], violence)

        # precision0 = torch.exp(self.log_vars[0])
        # precision0 = torch.exp(-self.log_vars[0])
        loss0 = loss0 

        # precision1 = torch.exp(self.log_vars[1])
        # precision1 = torch.exp(-self.log_vars[1])
        loss1 = loss1

        # precision2 = torch.exp(self.log_vars[2])
        # precision2 = torch.exp(-self.log_vars[2])
        loss2 = loss2 
        ### 试精度加权
        
        return loss0+loss1+loss2

class SingleLossWrapper(nn.Module):
    def __init__(self):
        super(SingleLossWrapper, self).__init__()
        # self.task_num = task_num

    def forward(self, preds, target):
        
        crossEntropy = nn.CrossEntropyLoss()
        loss0 = crossEntropy(preds, target)
        return loss0



    