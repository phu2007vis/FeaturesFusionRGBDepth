import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import resources.utils.heatmap_dataset as dsl
import resources.utils.pose_dataset as pose_dsl
import random
import datetime
import eval as ev
import yaml
import logging.handlers
import matplotlib.pyplot as plt
from resources.utils import *
from resources import get_model



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
def evaluate(model,model_name,dataloader,loss_fn,steps,class_info,ep,device = 'cuda',pbar = True):
    model.eval()
    logits = []
    current_loss = 0
    with torch.no_grad():
        
        data_iter = tqdm(dataloader,desc = "Evaluating") if pbar else dataloader
     
        for data in data_iter:
            
            if model_name != 'lstm':
                inputs, labels = data
                inputs = inputs.to(device)
            else:
                x_time,x_spatial ,labels = data
                inputs = (x_time.to(device),x_spatial.to(device))
            
            labels = labels.to(device)
            
            
            per_frame_logits = model(inputs)
            current_loss += loss_fn(per_frame_logits,labels).cpu().item()
            logit = per_frame_logits.max(-1)[1].cpu().numpy()
            labels = (labels.to(device).cpu().max(1)[1]).numpy()
                
            for i in range(len(logit)):
                logits.append([logit[i], labels[i]])
        
        elog.evaluate('val',steps,logits,class_info)
    current_loss = current_loss/(len(dataloader))
    elog.add_valid_loss(current_loss)
    elog.save_confusion_matrix("valid",ep, logits,class_info)
    
    
    model.train()
    return current_loss

def run(
        model_name,
        init_lr,
        max_steps,
        device,
        root,
        batch_size,
        n_frames,
        num_workers,
        evaluate_frequently,
        num_gradient_per_update,
        pretrained_path,
        learnig_scheduler_gammar,
        learnig_scheduler_step,
        seed = 42,
        cache=None,
        elog=None,
        name='i3d-rgb',
        num_keypoints = None,
        **kwargs):
    
    loss_fn = nn.CrossEntropyLoss()
    
    HEIGHT = 224
    WIDTH = 224
    if model_name != 'lstm':
        dataset = dsl.DSL(root, height=HEIGHT, width=WIDTH, n_frames=n_frames,random_seed=seed, cache_folder=cache)
    else:
        dataset = pose_dsl.DSL(root, n_frames=n_frames,random_seed=seed, cache_folder=cache)
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
        
    train_ds = dataset.get_generator(train_filter,mode = "train",spatial_augument = spatial_augument,**kwargs)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size,  num_workers=num_workers, pin_memory=True)

    val_ds = dataset.get_generator(val_filter,mode = "valid",**kwargs)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size,  num_workers=num_workers, pin_memory=True)

    test_ds = dataset.get_generator(test_filter,mode = "valid",**kwargs)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size,  num_workers=num_workers, pin_memory=True)
    
    dataloaders = {'train': train_dl, 'val': val_dl, 'test': test_dl}
    num_classes = len(dataset.get_classes())

  
  

    model = get_model(model_name,num_classes,num_keypoints = num_keypoints,n_frames = n_frames,**kwargs)
    model.to(device)
    
    if len(pretrained_path):
        model_state_dict= torch.load(pretrained_path,map_location=device)
        model.load_state_dict(model_state_dict)
 

    if True:
        for phase in ['train','val','test']:

            if phase == 'test':
                model.eval()
                current_valid_loss = evaluate(model,model_name,dataloaders[phase],loss_fn,0,class_info,ep = 0,device=device)
                print(f"Test loss: ",round(current_valid_loss,2))
                
                
               
    
    #save model
    torch.save(model.module.state_dict(), save_model+f'last.pt')
    plt.figure(clear=True)





if __name__ == "__main__":
    # need to add argparse
    parser = argparse.ArgumentParser()
    # model name s3d or i3d
    parser.add_argument("--model_name",type=str,default="i3d",help='i3d or s3d or lstm')
    parser.add_argument("--pretrained",type=str,default='/work/21013187/SignLanguageRGBD/all_code/results/i3d-72/14-13-47-27/i3d-72_best.pt')
    parser.add_argument("--device",type=str,default="cuda")
    parser.add_argument('-r', '--root', type=str, help='root directory of the dataset', default=r"/work/21013187/SignLanguageRGBD/data/ver2_all_rgb_only")
    parser.add_argument('--learnig_scheduler_gammar',type=float,default=0.7 ,help='decrease the learning rate by 0.6')
    parser.add_argument('--learnig_scheduler_step',type=int ,default=15)
    parser.add_argument('-n', '--n_frames', type=int, help='n frame', default= 72)
    parser.add_argument( '--num_keypoints', type=int, help='just for lstm', default= 66)
    
    parser.add_argument('-c', '--cache', type=str, help='cache directory', default=None)
    parser.add_argument('--seed', type=int, help='seed', default=42)
    parser.add_argument('--a_config', type=str, help='spatial augumentation config', default="train_sh/config/spatial_augument_config.yaml")
    parser.add_argument('--lr',type=float,default =0.001, help='init learning rate')
    parser.add_argument('--epochs', type=int, help='number of training epochs', default=270)
    parser.add_argument('--batch_size', type=int, help='batch_size', default=9)
    parser.add_argument('--num_workers', type=int, help='number of cpu load data', default=8)
    parser.add_argument('--evaluate_frequently', type=int, help='number of cpu load data', default=200)
    parser.add_argument('--num_gradient_per_update', type=int, help='number of cpu load data', default=25)
    parser.add_argument('--fintuning',type = int, default = -1)
  
    args = parser.parse_args()
    root = args.root
    
    model_name = args.model_name
  
    n_frames = args.n_frames
    pretrained_path = args.pretrained
    
    num_gradient_per_update = args.num_gradient_per_update
    evaluate_frequently = (args.evaluate_frequently // num_gradient_per_update)*num_gradient_per_update
    
    learnig_scheduler_gammar = args.learnig_scheduler_gammar
    learnig_scheduler_step = args.learnig_scheduler_step
    
    print(f"Evaluate frequently: {evaluate_frequently}")
    print(f"Num gradient per update: {num_gradient_per_update}")
    
    name = f"eval_{model_name}-{n_frames}"
    elog = ev.Eval(run_name=name)
    
    run(
        model_name=model_name,
        device = args.device, 
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
        num_gradient_per_update = num_gradient_per_update,
        pretrained_path = pretrained_path, 
        learnig_scheduler_gammar = learnig_scheduler_gammar,
        learnig_scheduler_step = learnig_scheduler_step,
        num_keypoints = args.num_keypoints,
        fintuning = args.fintuning
        )
