
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import WarmUpLR, evaInfo
from options import args
import os
from torch import optim
from dataset import ic_dataset
from ICNet import ICNet


def train(epoch):
    model.train()
    loss1_sum = 0
    loss2_sum = 0
    loss_tot = 0
    for batch_index, (image,label,_) in enumerate(trainDataLoader):        
        image = image.to(device)
        label = label.to(device)
        label = label / 9
        Opmimizer.zero_grad()
        score1, cly_map = model(image)
        score2 = cly_map.mean(axis = (1,2,3))
        loss1 = loss_function(score1,label)
        loss2 = loss_function(score2,label)
        loss = 0.9*loss1 + 0.1*loss2
        loss.backward()
        Opmimizer.step()

        loss1_sum += loss1
        loss2_sum += loss2
        loss_tot += loss
    
    print(f'Epoch: {epoch} \t Total Loss: {loss_tot:.4f} \t Loss1: {loss1_sum:.4f} \t Loss2: {loss2_sum:.4f}')

        

def evaluation():
    model.eval()
    all_scores = []
    all_labels = []
    for (image, label, _) in valDataLoader:
        image = image.to(device)
        label = label.to(device)
        label = label / 9
        with torch.no_grad():
            score, _= model(image)
            all_scores += score * 9
            all_labels += label * 9
    score = torch.stack(all_scores, dim = 0)
    label = torch.stack(all_labels, dim = 0)
    info = evaInfo(score = score, label = label)
    print(info + '\n')




if __name__ == "__main__":
    
    trainTransform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size), interpolation = InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[173.10, 175.12, 177.00], std=[94.64, 89.89, 89.64])
    ])

    valTransform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size), interpolation = InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[173.10, 175.12, 177.00], std=[94.64, 89.89, 89.64])
    ])
  
    trainDataset = ic_dataset(
        txt_path ="../dataset_default/meta/train.txt",
        img_path = "",
        transform = trainTransform
    )
    
    trainDataLoader = DataLoader(trainDataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=True
                             )

    valDataset = ic_dataset(
        txt_path= "../dataset_default/meta/val.txt",
        img_path = "",
        transform=valTransform
    )
    
    valDataLoader = DataLoader(valDataset,
                            batch_size=args.batch_size,
                               num_workers=args.num_workers,
                            shuffle=False
                            )
    if not os.path.exists(args.ck_save_dir):
        os.mkdir(args.ck_save_dir)
    
    model = ICNet() 
    
    device = torch.device("cuda:{}".format(args.gpu_id))
    model.to(device)

    loss_function = nn.MSELoss()
    
    # optimize
    params = model.parameters()
    Opmimizer = optim.SGD(params, lr =args.lr,momentum=0.9,weight_decay=args.weight_decay)
    Scheduler = optim.lr_scheduler.MultiStepLR(Opmimizer,milestones=args.milestone,gamma = args.lr_decay_rate)
    iter_per_epoch = len(trainDataLoader)
    
    # running
    for epoch in range(1, args.epoch+1):
        train(epoch)
        evaluation()
        #torch.save(model.state_dict(), os.path.join(args.ck_save_dir,'ck_{}.pth'.format(epoch)))

    








