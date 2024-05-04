import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import resources.utils.rgb_dataset as dsl
import random
import numpy as np
import datetime
from resources.i3d.pytorch_i3d import InceptionI3d
import eval as ev
os.makedirs('logs',exist_ok=True)
import logging.handlers

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

def run(init_lr=0.1, max_steps=200000, device = "cuda", root="/work/21010294/DepthData/OutputSplitAbsoluteVer2/", batch_size=8,
        name='i3d-rgb', n_frames=200, elog=None,seed=42, cache=None):
    loss_fn = nn.CrossEntropyLoss()
    random.seed(seed)
    HEIGHT = 224
    WIDTH = 224
    dataset = dsl.DSL(root, height=HEIGHT, width=WIDTH, n_frames=n_frames,random_seed=seed, cache_folder=cache)
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

    train_filter = dataset.filter(persons=train_persons)
    val_filter = dataset.filter(persons=val_persons)
    test_filter = dataset.filter(persons=test_persons)

    print(f"Train set size: {len(train_filter['class'])}")
    print(f"Validation set size: {len(val_filter['class'])}")
    print(f"Test set size: {len(test_filter['class'])}")
    


    train_ds = dataset.get_generator(train_filter,mode = "train")
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,  num_workers=2, pin_memory=True)

    val_ds = dataset.get_generator(val_filter,mode = "valid")
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,  num_workers=0, pin_memory=True)

    test_ds = dataset.get_generator(test_filter,mode = "test")
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,  num_workers=0, pin_memory=True)

    dataloaders = {'train': train_dl, 'val': val_dl, 'test': test_dl}

    num_classes = len(dataset.get_classes())
 
    model = InceptionI3d(400, in_channels=3)
    #load pretrained model
    # model.load_state_dict(torch.load('models/rgb_imagenet.pt'))
    model.replace_logits(num_classes)
    model.to(device)
    model = nn.DataParallel(model)
    print(f"Train on {device}")
    lr = init_lr
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0000001)
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
                model.train()
            else:
                model.train()  # Set model to evaluate mode
                
            tot_loss = 0.0
            num_iter = 0
            optimizer.zero_grad()

            if phase == 'train':
                for data in dataloaders[phase]:
                    num_iter += 1
                    # get the inputs
                    inputs, labels = data

                    # wrap them in Variable
                    inputs = inputs.to(device)
                    t = inputs.size(2)
                    labels = labels.to(device)
                    import pdb;pdb.set_trace()
                    per_frame_logits = model(inputs)
                    # upsample to input size

                    loss = loss_fn(per_frame_logits,labels)/num_steps_per_update
                    tot_loss += loss.data.item()
                    loss.backward()
                    if num_iter == num_steps_per_update and phase == 'train':
                        steps += 1
                        pbar.update(1)
                        num_iter = 0
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_sched.step()
                        
                        if steps % 10 == 0:
                            elog.add_loss_entry(f"{steps},{tot_loss}")
                            # save model
                            torch.save(model.module.state_dict(), save_model+str(steps).zfill(6)+'.pt')
                            #add checkpoint
                            checkpoints.append(save_model+str(steps).zfill(6)+'.pt')
                            if len(checkpoints) > 3:
                                os.remove(checkpoints.pop(0))
                        tot_loss =  0

            if phase == 'val':
              
                print("Evaluating")
                model.eval()
                logits = []
                for data in dataloaders[phase]:
                    inputs, labels = data
                    inputs = inputs.to(device)
                    labels = (inputs.to(device).cpu().max(1)[1]).numpy()
                    
                    per_frame_logits = model(inputs)
                    logit = nn.functional.softmax(per_frame_logits,dim = 1)
                    logit = logit.max(1)[1].cpu().numpy()
                    for i in range(len(logit)):
                        logits.append([predicted[i], labels[i]])
                        
                elog.evaluate(phase,steps,logits, dataset.get_classes())
                model.train()
    
    pbar.close()
    #save model
    torch.save(model.module.state_dict(), save_model+f'i3d_train_final.pt')
    #test
    model.eval()
    print("Testing")
    logits = []
    for data in dataloaders['test']:
        inputs, labels = data
        inputs = Variable(inputs.cuda())
        t = inputs.size(2)
        labels = Variable(labels.cuda())
        per_frame_logits = model(inputs)
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
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument('-r', '--root', type=str, help='root directory of the dataset', default=r"/work/21013187/SignLanguageRGBD/ViSLver2/Processed")
    parser.add_argument('-n', '--n_frames', type=int, help='n frame', default=10)
    parser.add_argument('-c', '--cache', type=str, help='cache directory', default=None)
    parser.add_argument('-s', '--seed', type=int, help='seed', default=42)
    
    args = parser.parse_args()
    root = args.root
    n_frames = args.n_frames


    name = f"i3d-train{n_frames}"
    elog = ev.Eval(run_name=name)
    run(device = args.device, root=root, name=name, n_frames=n_frames, elog=elog,seed=args.seed,cache=args.cache)
