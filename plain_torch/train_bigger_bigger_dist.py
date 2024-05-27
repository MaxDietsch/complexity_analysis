import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from options import args
from utils import evaInfo
import os
from torch import optim
from dataset import ic_dataset
from ICNet_bigger_bigger import ICNet
from torch.nn import functional as F

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    dist.init_process_group('nccl', rank = rank, world_size = world_size)

def cleanup():
    os.environ['MASTER_ADDR'] = 'localhost'  # or the IP address of the master node
    os.environ['MASTER_PORT'] = '12355'
    dist.destroy_process_group()


def main(args):
    rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
    world_size = int(os.environ.get('WORLD_SIZE', args.world_size))
    setup(rank, world_size)
    
    trainTransform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[173.10, 175.12, 177.00], std=[94.64, 89.89, 89.64])
    ])

    valTransform = transforms.Compose([
    transforms.Resize((args.image_size, args.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[173.10, 175.12, 177.00], std=[94.64, 89.89, 89.64])
    ])
    

    trainDataset = ic_dataset(
        txt_path ="../dataset_default/meta/train.txt",
        img_path = "",
        transform = trainTransform
    )
    
    trainSampler = DistributedSampler(trainDataset, num_replicas = world_size, rank = rank)

    
    trainDataLoader = DataLoader(trainDataset,
                                batch_size = int(args.batch_size / world_size),
                                num_workers = args.num_workers,
                                sampler = trainSampler
                                )

    valDataset = ic_dataset(
        txt_path= "../dataset_default/meta/val.txt",
        img_path = "",
        transform=valTransform
    )

    valSampler = DistributedSampler(valDataset, num_replicas = world_size, rank = rank)
    
    valDataLoader = DataLoader(valDataset,
                                batch_size = int(args.batch_size / world_size),
                                num_workers = args.num_workers,
                                sampler = valSampler
                                )


    if not os.path.exists(args.ck_save_dir):
        os.mkdir(args.ck_save_dir)
    
    model = ICNet(args.image_size, 256).to(rank)
    model = DDP(model, device_ids = [rank])
    

    loss_function = nn.MSELoss()
    
    # optimize
    params = model.parameters()
    Opmimizer = optim.SGD(params, lr =args.lr, momentum=0.9, weight_decay=args.weight_decay)
    Scheduler = optim.lr_scheduler.MultiStepLR(Opmimizer,milestones=args.milestone,gamma = args.lr_decay_rate)
    
    #loop 
    for epoch in range(1, args.epoch+1):
        model.train()
        loss1_sum = 0
        loss2_sum = 0
        loss_tot = 0
        #train
        for batch_index, (image,label,_) in enumerate(trainDataLoader):        
            image = image.to(rank)
            label = label.to(rank)
            label = label / 9 
            Opmimizer.zero_grad()
            score1, cly_map = model(image)
            #print(score1)
            #print(label)
            score2 = cly_map.mean(axis = (1,2,3))
            loss1 = loss_function(score1,label)
            loss2 = loss_function(score2,label)
            #loss = 0.9*loss1 + 0.1*loss2
            loss = 0.7 * loss1 + 0.3 * loss2
            loss.backward()
            Opmimizer.step()

            loss1_sum += loss1
            loss2_sum += loss2
            loss_tot += loss
    
        print(f'Rank: {rank}: Epoch: {epoch} \t Total Loss: {loss_tot:.4f} \t Loss1: {loss1_sum:.4f} \t Loss2: {loss2_sum:.4f}')
        
        #eval
        #model.eval()
        all_scores = []
        all_labels = []
        for (image, label, _) in valDataLoader:
            image = image.to(rank)
            label = label.to(rank)
            label = label / 9
            with torch.no_grad():
                score, _= model(image)

                all_scores += list(torch.split(score, 1, dim = 0))
                all_labels += list(torch.split(label, 1, dim = 0))
        
        all_scores = torch.stack(all_scores)
        all_labels = torch.stack(all_labels)

        #print(all_scores)
        #print(all_labels)
                
        info = evaInfo(score = score, label = label)
        print(info + '\n')

    
    cleanup()
    torch.save(model.state_dict(), os.path.join(args.ck_save_dir, 'ck_bigger_bigger{}_bs{}_is{}_lr{}.pth'.format(epoch, args.batch_size, args.image_size, args.lr)))

    
if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    print(f'Found devices: {world_size}')
    main(args)







