import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
#os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2,3'
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from tqdm import tqdm
import torchvision
from torchvision import transforms
import DSL.torch as dsl
import random
import numpy as np
import time
import datetime
from pytorch_i3d import InceptionI3d





import eval as ev
os.makedirs('logs',exist_ok=True)
import logging.handlers
# Set up logging
log_filename = datetime.datetime.now().strftime("sc_%d_%m_%H_%M_%S.log")
log_filepath = os.path.join("logs", log_filename)



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(log_filepath, maxBytes=(1048576*5), backupCount=7),
        logging.StreamHandler(sys.stdout)
    ],force=True
)

def handle_exception(exc_type, exc_value, exc_traceback):
    # Custom exception handling logic here
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    # Call the default handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

sys.excepthook = handle_exception

def run(init_lr=0.1, max_steps=200000, mode='rgb', root="/work/21010294/DepthData/OutputSplitAbsoluteVer2/", batch_size=1,
        name='i3d-rgb', nframe=200, elog=None,seed=42, cache=None):
    # setup dataset
    # train_transforms = transforms.Compose([videotransforms.RandomCrop(224),
    #                                        videotransforms.RandomHorizontalFlip(),
    # ])
    random.seed(seed)
    HEIGHT = 224
    WIDTH = 224
    dataset = dsl.DSL(root, height=HEIGHT, width=WIDTH, n_frames=200,random_seed=seed, cache_folder=cache)
    person_list = dataset.get_persons()
    random.shuffle(person_list)
    val_index = int(len(person_list) * 0.7)
    test_index = int(len(person_list) * 0.8)
    train_persons = person_list[:val_index]
    val_persons =   person_list[val_index:test_index]
    test_persons =  person_list[test_index:]
    save_model = elog.get_path() + f"/{name}_"
    print(f"Train: {len(train_persons)}",
        f" Val: {len(val_persons)}",
        f" Test: {len(test_persons)}")
    fullset = dataset.filter()
    #train_filter,val_filter,test_filter = dataset.divide_filter(fullset,[0.8,0.1,0.1])

    


    train_filter = dataset.filter(persons=train_persons)
    val_filter = dataset.filter(persons=val_persons)
    test_filter = dataset.filter(persons=test_persons)

    print(f"Train set size: {len(train_filter['class'])}")
    print(f"Validation set size: {len(val_filter['class'])}")
    print(f"Test set size: {len(test_filter['class'])}")
    


    train_ds = dataset.get_generator(train_filter,output=mode)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,  num_workers=2, pin_memory=True)

    val_ds = dataset.get_generator(val_filter,output=mode)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,  num_workers=0, pin_memory=True)

    test_ds = dataset.get_generator(test_filter,output=mode)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,  num_workers=0, pin_memory=True)

    dataloaders = {'train': train_dl, 'val': val_dl, 'test': test_dl}

    num_classes = len(dataset.get_classes())
    # setup the model
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load('models/flow_imagenet.pt'))
        print("Loaded pretrained model for flow")
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load('models/rgb_imagenet.pt'))
        print("Loaded pretrained model for rgb")
    i3d.replace_logits(num_classes)
    #i3d.load_state_dict(torch.load('test.pt000010.pt'))
    i3d.cuda()
    i3d = nn.DataParallel(i3d)

    lr = init_lr
    optimizer = optim.SGD(i3d.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
    lr_sched = optim.lr_scheduler.MultiStepLR(optimizer, [300, 1000])
    checkpoints = []

    num_steps_per_update = 4 # accum gradient
    steps = 0
    # train it
    pbar = tqdm(total=max_steps)
    while steps < max_steps:#for epoch in range(num_epochs):
        print( 'Step {}/{}'.format(steps, max_steps))
        print( '-' * 10)
        


        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                i3d.train(True)
            else:
                i3d.train(False)  # Set model to evaluate mode
                
            tot_loss = 0.0
            tot_loc_loss = 0.0
            tot_cls_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()
            
            # Iterate over data.
            if phase == 'train':
                for data in dataloaders[phase]:
                    num_iter += 1
                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    inputs = Variable(inputs.cuda())
                    t = inputs.size(2)
                    labels = Variable(labels.cuda())

                    per_frame_logits = i3d(inputs)
                    # upsample to input size
                    per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')

                    # compute localization loss
                    loc_loss = F.binary_cross_entropy_with_logits(per_frame_logits, labels)
                    tot_loc_loss += loc_loss.data.item()

                    # compute classification loss (with max-pooling along time B x C x T)
                    cls_loss = F.binary_cross_entropy_with_logits(torch.max(per_frame_logits, dim=2)[0], torch.max(labels, dim=2)[0])
                    tot_cls_loss += cls_loss.data.item()

                    loss = (0.5*loc_loss + 0.5*cls_loss)/num_steps_per_update
                    tot_loss += loss.data.item()
                    loss.backward()
                    print(f"Loss: {loss.data.item()}, Loc Loss: {loc_loss.data.item()}, Cls Loss: {cls_loss.data.item()},tot_loss: {tot_loss}")
                    if num_iter == num_steps_per_update and phase == 'train':
                        steps += 1
                        pbar.update(1)
                        num_iter = 0
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_sched.step()
                        if steps % 10 == 0:
                            print('{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/(10*num_steps_per_update), tot_cls_loss/(10*num_steps_per_update), tot_loss/10))
                            elog.add_loss_entry(f"{steps},{tot_loss/10},{tot_loc_loss/(10*num_steps_per_update)},{tot_cls_loss/(10*num_steps_per_update)},{tot_loss/10}")
                            # save model
                            torch.save(i3d.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                            #add checkpoint
                            checkpoints.append(save_model+str(steps).zfill(6)+'.pt')
                            if len(checkpoints) > 3:
                                os.remove(checkpoints.pop(0))

                            tot_loss = tot_loc_loss = tot_cls_loss = 0.
            if phase == 'val':
                #print( '{} Loc Loss: {:.4f} Cls Loss: {:.4f} Tot Loss: {:.4f}'.format(phase, tot_loc_loss/num_iter, tot_cls_loss/num_iter, (tot_loss*num_steps_per_update)/num_iter) )
                #evaluate and print accuracy
                print("Evaluating")
                i3d.eval()
                logits = []
                for data in dataloaders[phase]:
                    inputs, labels = data
                    inputs = Variable(inputs.cuda())
                    t = inputs.size(2)
                    labels = Variable(labels.cuda())
                    per_frame_logits = i3d(inputs)
                    # upsample to input size
                    per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')
                    logit = per_frame_logits[0].data.cpu().numpy()
                    classif = torch.max(per_frame_logits, dim=2)[0]
                    classif = classif[0].data.cpu().numpy()
                    label = torch.max(labels, dim=2)[0]
                    labels = label[0].data.cpu().numpy()
                    labels = np.argmax(labels)
                    predicted = np.argmax(classif)

                    logits.append([logit, classif,predicted, labels])
                elog.evaluate(phase,steps,logits, dataset.get_classes())
                i3d.train(True)
    
    pbar.close()
    #save model
    torch.save(i3d.module.state_dict(), save_model+f'i3d_{mode}_final.pt')
    #test
    i3d.eval()
    print("Testing")
    logits = []
    for data in dataloaders['test']:
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        t = inputs.size(2)
        labels = Variable(labels.cuda())
        per_frame_logits = i3d(inputs)
        # upsample to input size
        per_frame_logits = F.upsample(per_frame_logits, t, mode='linear')
        logit = per_frame_logits[0].data.cpu().numpy()
        classif = torch.max(per_frame_logits, dim=2)[0]
        classif = classif[0].data.cpu().numpy()
        label = torch.max(labels, dim=2)[0]
        labels = label[0].data.cpu().numpy()
        labels = np.argmax(labels)
        predicted = np.argmax(classif)

        logits.append([logit, classif,predicted, labels])
    elog.evaluate('test',steps,logits,dataset.get_classes())



if __name__ == '__main__':
    # need to add argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, help='rgb or flow', default='rgb')
    parser.add_argument('-r', '--root', type=str, help='root directory of the dataset', default=r'D:\2023-2024\Project\Realsense\output_split')
    parser.add_argument('-n', '--nframe', type=int, help='n frame', default=200)
    parser.add_argument('-c', '--cache', type=str, help='cache directory', default=None)
    parser.add_argument('-s', '--seed', type=int, help='seed', default=42)
    

    args = parser.parse_args()
    mode = args.mode
    root = args.root
    nframe = args.nframe


    name = f"i3d-{mode}{nframe}"
    elog = ev.Eval(run_name=name)
    run(mode=args.mode, root=root, name=name, nframe=nframe, elog=elog,seed=args.seed,cache=args.cache)
