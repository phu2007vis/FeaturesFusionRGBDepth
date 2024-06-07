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
import matplotlib.pyplot as plt
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
def evaluate(model,dataloader,loss_fn,steps,class_info,device = 'cuda',pbar = True):
    model.eval()
    logits = []
    current_loss = 0
    with torch.no_grad():
        
        data_iter = tqdm(dataloader,desc = "Evaluating") if pbar else dataloader
     
        for data in data_iter:
            
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            per_frame_logits = model(inputs)
            current_loss += loss_fn(per_frame_logits,labels).cpu().item()
            logit = per_frame_logits.max(-1)[1].cpu().numpy()
            labels = (labels.to(device).cpu().max(1)[1]).numpy()
                
            for i in range(len(logit)):
                logits.append([logit[i], labels[i]])
        
        elog.evaluate('val',steps,logits,class_info)
    current_loss = current_loss/(len(dataloader))
    
    
    model.train()
    return current_loss
    
def run(init_lr,
        max_steps,
        device,
        root,
        batch_size,
        n_frames,
        num_workers,
        evaluate_frequently,
        num_gradient_per_update,
        seed = 42,
        cache=None,
        elog=None,
        name='i3d-rgb'
        ):
    
    loss_fn = nn.CrossEntropyLoss()
    
    HEIGHT = 224
    WIDTH = 224
    
    dataset = dsl.DSL(root, height=HEIGHT, width=WIDTH, n_frames=n_frames,random_seed=seed, cache_folder=cache)
    class_info = dataset.get_classes()
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
    train_loss = []
    valid_loss = []
    best_valid_loss = 99999
    for epoch in range(max_steps):#for epoch in range(num_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            
            if phase == 'train':
               
                #get current learning rate
                lr = optimizer.param_groups[0]['lr']
                optimizer.zero_grad()
                model.train()
                
                #reset parameter
                tot_loss = 0.0
                num_iter = 0
                
                #create process bar
                pbar = tqdm(enumerate(dataloaders[phase]),total=len(dataloaders[phase]))
                
                for index , (inputs, labels) in pbar:
                    
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
                        train_loss.append(tot_loss)
                        # reset parameters
                        num_iter = 0
                        tot_loss = 0
                        if (index+1) % evaluate_frequently == 0:
                            current_valid_loss = evaluate(model,dataloaders['val'],loss_fn,steps,class_info,device=device,pbar=False)
                            valid_loss.append(current_valid_loss)
                            pbar.set_postfix_str(f"Valid loss: {round(current_valid_loss,2)}")
                            
                            if current_valid_loss < best_valid_loss:
                                torch.save(model.state_dict(),save_model+"best.pt")
                                best_valid_loss = current_valid_loss
                            break
                        
                torch.save(model.state_dict(),save_model+"last.pt")
                        
     
           # torch.save(model.module.state_dict(), save_model+'last.pt')
            if phase == 'val':
                current_valid_loss = evaluate(model,dataloaders['val'],loss_fn,steps,class_info,device=device)
                valid_loss.append(current_valid_loss)
                
                print(f"Val loss: ",round(current_valid_loss,2))
                
                if current_valid_loss < best_valid_loss:
                    torch.save(model.state_dict(),save_model+"best.pt")
                    best_valid_loss = current_valid_loss
        break
                    
                
               
    
    #save model
    torch.save(model.module.state_dict(), save_model+f'last.pt')
    plt.figure(clear=True)
    plt.plot(train_loss)
    plt.ylabel("Loss")
    plt.xlabel("Batch")
    plt.title("Train loss")
    plt.savefig(save_model+"train_loss.png")
    plt.close()
    plt.figure(clear=True)
    plt.plot(valid_loss)
    plt.ylabel("Loss")
    plt.xlabel("Batch")
    plt.title("Validation loss")
    plt.savefig(save_model+"val_loss.png")
    plt.close()




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
    parser.add_argument('--evaluate_frequently', type=int, help='number of cpu load data', default=4)
    parser.add_argument('--num_gradient_per_update', type=int, help='number of cpu load data', default=4)
   

    args = parser.parse_args()
    root = args.root
    n_frames = args.n_frames
    num_gradient_per_update = args.num_gradient_per_update
    evaluate_frequently = (args.evaluate_frequently // num_gradient_per_update)*num_gradient_per_update
    print(f"Evaluate frequently: {evaluate_frequently}")
    print(f"Num gradient per update: {num_gradient_per_update}")
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
        num_workers = args.num_workers,
        evaluate_frequently = evaluate_frequently,
        num_gradient_per_update = num_gradient_per_update
        )
