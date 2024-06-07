import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import resources.utils.rgb_dataset as dsl
import random
import datetime
from resources.i3d.pytorch_i3d import InceptionI3d
import eval as ev
import yaml
import logging.handlers
from resources.utils import *



# save infomation during training
os.makedirs('logs',exist_ok=True)
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
# print some thinh when error 
def handle_exception(exc_type, exc_value, exc_traceback):
    # Custom exception handling logic here
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    # Call the default handler
    sys.__excepthook__(exc_type, exc_value, exc_traceback)
sys.excepthook = handle_exception

def run(init_lr,
        max_steps,
        device,
        root,
        batch_size,
        n_frames,
        num_workers,
        seed = 42,
        cache=None,
        elog=None,
        name='i3d-rgb'):
    
    loss_fn = nn.CrossEntropyLoss()
    
    HEIGHT = 224
    WIDTH = 224
    
    dataset = dsl.DSL(root, height=HEIGHT, width=WIDTH, n_frames=n_frames,random_seed=seed, cache_folder=cache)
    person_list = dataset.get_persons()
    
    random.seed(seed)
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
    
    with open(args.a_config,'r') as f:
        try:
            spatial_augument = yaml.safe_load(f).get("augument")
        except:
            spatial_augument = None
        
    train_ds = dataset.get_generator(train_filter,mode = "train",spatial_augument = spatial_augument)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,  num_workers=num_workers, pin_memory=True)

    val_ds = dataset.get_generator(val_filter,mode = "valid")
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,  num_workers=num_workers, pin_memory=True)

    test_ds = dataset.get_generator(test_filter,mode = "valid")
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,  num_workers=num_workers, pin_memory=True)
    
    dataloaders = {'train': train_dl, 'val': val_dl, 'test': test_dl}
    num_classes = len(dataset.get_classes())
    
    #turn on this for visualize
    # visualize_rgb(train_dl,"visualize",percent_visualize=0.5)
    # exit()
    
    model = InceptionI3d(400, in_channels=3)
    model.replace_logits(num_classes)
    model.to(device)
    model = nn.DataParallel(model)
    print(f"Train on {device}")
    
    lr = init_lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0000001)

    num_steps_per_update = 4 # accum gradient
    steps = 0
    
    for epoch in range(max_steps):#for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            
            if phase == 'train':
                continue
                #get current learning rate
                lr = optimizer.param_groups[0]['lr']
                optimizer.zero_grad()
                model.train()
                
                #reset parameter
                tot_loss = 0.0
                num_iter = 0
                
                #create process bar
                pbar = tqdm(dataloaders[phase],total=len(dataloaders[phase]))
                
                for inputs, labels in pbar:
                    
                    num_iter += 1

                    # move to device ('cpu' or 'gpu' - 'cuda' )
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    
                    #forward
                    per_frame_logits = model(inputs)
            
                    #caculate loss
                    loss = loss_fn(per_frame_logits,labels)/num_steps_per_update
                    
                    #caculate gradient 
                    loss.backward()
                    
                    #convert to float and add to total loss
                    tot_loss += loss.data.item()
                    
                    # update each num_steps_per_update batch
                    if num_iter == num_steps_per_update :
                        
                        #update weight
                        optimizer.step()
                        optimizer.zero_grad()
                        
                        #print to screen
                        info  =f"{epoch}/{max_steps} , lr : {lr} , train loss : {tot_loss}" 
                        pbar.set_description(info)
                        pbar.set_postfix()
                        
                        # reset parameters
                        num_iter = 0
                        tot_loss = 0
                        
     
           # torch.save(model.module.state_dict(), save_model+'last.pt')
            if phase == 'val':
         
                
                model.eval()
                logits = []
                with torch.no_grad():
                    for data in tqdm(dataloaders[phase],desc = "Evaluating"):
                        inputs, labels = data
                        inputs = inputs.to(device)
                        labels = (labels.to(device).cpu().max(1)[1]).numpy()
                        
                        per_frame_logits = model(inputs)
                      
                        logit = per_frame_logits.max(-1)[1].cpu().numpy()
                       
                            
                        for i in range(len(logit)):
                            logits.append([logit[i], labels[i]])
                    
                    elog.evaluate(phase,steps,logits, dataset.get_classes())
                model.train()
    
    #save model
    torch.save(model.module.state_dict(), save_model+f'i3d_train_final.pt')



if __name__ == '__main__':
    # need to add argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument('-r', '--root', type=str, help='root directory of the dataset', default=r"/work/21013187/SignLanguageRGBD/ViSLver2/Processed")
    parser.add_argument('-n', '--n_frames', type=int, help='n frame', default= 72)
    parser.add_argument('-c', '--cache', type=str, help='cache directory', default=None)
    parser.add_argument('--seed', type=int, help='seed', default=42)
    parser.add_argument('--a_config', type=str, help='spatial augumentation config', default="train_sh/config/spatial_augument_config.yaml")
    parser.add_argument('--lr',type=float,default =0.001, help='init learning rate')
    parser.add_argument('--epochs', type=int, help='number of training epochs', default=1000)
    parser.add_argument('--batch_size', type=int, help='batch_size', default=9)
    parser.add_argument('--num_workers', type=int, help='number of cpu load data', default=8)

    args = parser.parse_args()
    root = args.root
    n_frames = args.n_frames
    
    name = f"i3d-train{n_frames}"
    elog = ev.Eval(run_name=name)
    run(device = args.device, 
        root=root, name=name,
        n_frames=n_frames,
        elog=elog,
        seed=args.seed,
        cache=args.cache,
        init_lr = args.lr,
        max_steps = args.epochs,
        batch_size = args.batch_size,
        num_workers = args.num_workers)
