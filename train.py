
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
        Opmimizer.zero_grad()
        score1, cly_map = model(image)
        score2 = cly_map.mean(axis = (1,2,3))
        loss1 = loss_function(score1,label)
        loss2 = loss_function(score2,label)
        loss = 0.9*loss1 + 0.1*loss2
        loss.backward()
        Opmimizer.step()
        if epoch <= args.warm:
            Warmup_scheduler.step()

        loss1_sum += loss1
        loss2_sum += loss2
        loss_tot += loss
    
    print(f'Epoch: {epoch} \t Total Loss: {loss_tot} \t Loss1: {loss1_sum} \t Loss2: {loss2_sum}')
    print('\n')

        

def evaluation():
    model.eval()
    all_scores = []
    all_labels = []
    for (image, label, _) in testDataLoader:
        image = image.to(device)
        label = label.to(device)
        with torch.no_grad():
            score, _= model(image)
            all_scores += score.tolist()
            all_labels += label.tolist()
    info = evaInfo(score=all_scores, label=all_labels)
    print(info + '\n')




if __name__ == "__main__":
    
    trainTransform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    testTransform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
  
    trainDataset = ic_dataset(
        txt_path ="../dataset_default/meta/train_min_min.txt",
        img_path = "train",
        transform = trainTransform
    )

    
    trainDataLoader = DataLoader(trainDataset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=True
                             )

    testDataset = ic_dataset(
        txt_path= "../dataset_default/meta/val.txt",
        img_path = "val",
        transform=testTransform
    )
    
    testDataLoader = DataLoader(testDataset,
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
    if args.warm > 0:
        Warmup_scheduler = WarmUpLR(Opmimizer,iter_per_epoch*args.warm)
    
    # running
    for epoch in range(1, args.epoch+1):
        train(epoch)
        if epoch > args.warm:
            Scheduler.step(epoch)
        #evaluation()
        #torch.save(model.state_dict(), os.path.join(args.ck_save_dir,'ck_{}.pth'.format(epoch)))

    








