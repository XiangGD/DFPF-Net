import argparse
import logging
import os
import pickle
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.modules.utils import _pair
from tqdm import tqdm

from sklearn.metrics import roc_curve, roc_auc_score, auc
from datasets.dataset_forensic import Forensic_dataset
from utils import test_single_sample, metric_img_score
from networks.vit_seg_modeling import ForensicTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

parser = argparse.ArgumentParser()
#dataset parameters
parser.add_argument('--root_path',type=str,default='../datasets', help='root dir for validation volume data') 
parser.add_argument('--dataset',  type=str, default='PS', help='experiment_name')
parser.add_argument('--test_data',  type=str, default='NIST2016', help='experiment_name')
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
parser.add_argument('--list_dir', type=str, default='./lists', help='list dir') 
#model parameters
parser.add_argument('--vit_name', type=str,default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,default=16, help='vit_patches_size, default is 16')
parser.add_argument('--max_epochs', type=int,default=70, help='maximum epoch number to train')
parser.add_argument('--ft_epochs', type=int,default=70, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,default=21, help='batch_size per gpu')
parser.add_argument('--n_skip', type=int,default=3, help='using number of skip-connect, default is num')
parser.add_argument('--seed', type=int,default=1234, help='random seed')
parser.add_argument('--img_size', type=int,default=256, help='input patch size of network input')
parser.add_argument('--lr_name', type=str,default='Customized', help='lr schedule type')
parser.add_argument('--optimizer', type=str,default='Adam', help='optimizier type，SGD or Adam')
parser.add_argument('--base_lr', type=float,  default=0.0001,help='segmentation network learning rate')
parser.add_argument('--split', type=str,default='fine_tuned',help='split train model or fine tune model')
parser.add_argument('--best_model', type=int,default=47,help='split train model or fine tune model')
# test result save parameters
parser.add_argument('--is_save', default= True, action="store_true", help='whether to save results during inference')

args = parser.parse_args()
  

def inference(args, model,split):
    db_test = Forensic_dataset(args, base_dir=args.root_path, list_dir=args.list_dir,
                           split=split, img_size=_pair(args.img_size))
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info("{} test iterations per epoch".format(len(testloader)))
    device = args.device
    model.to(device)
    model.eval()
    #print(args.premask_path)

    mean_f1, mean_iou, mean_auc, mean_precision, mean_recall = 0, 0, 0, 0, 0

    num_samples = len(testloader)
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        _, _, h, w = sampled_batch["image"].size()
        assert (h,w)==_pair(args.img_size)
        image, label = sampled_batch["image"], sampled_batch["label"]
        cls, sample_name = sampled_batch['cls'][0], sampled_batch['name'][0]

        image, label = image.to(device), label.to(device)
        f1, iou, precision, recall, pixel_auc, _ = test_single_sample(image, label, model,
                                                           num_class=args.num_classes,
                                                           test_save_path=args.premask_path,
                                                           case=sample_name
                                                           )
        
      
        mean_f1 += f1
        mean_iou += iou
        mean_precision += precision
        mean_recall += recall
        mean_auc += pixel_auc
       

        #if split == 'test':
        #logging.info('%s-> single_f1: %f, single_iou: %f, single_p: %f, single_r: %f, single_auc: %f, single_score: %f' \
         #                % (sample_name, f1, iou, precision, recall, pixel_auc, max_score))
        
  
    fpr, tpr, thresholds = roc_curve((np.array(classes) > 0).astype(np.int), (np.array(scores)>0.5).astype(np.int), pos_label=1)
    
    with open(os.path.join(args.graph_path, 'epo'+str(args.best_model)+'_roc.pkl'), 'wb') as f:
        pickle.dump({'fpr': fpr, 'tpr': tpr, 'th': thresholds}, f)
        logging.info("roc save at %s" % (os.path.join(args.graph_path, 'epo'+str(args.best_model)+'_roc.pkl')))

    mean_f1        = mean_f1 / num_samples
    mean_iou       = mean_iou / num_samples
    mean_precision = mean_precision / num_samples
    mean_recall    = mean_recall / num_samples
    mean_auc       = mean_auc / num_samples

    logging.info("pixel-mean_f1: %.4f,pixel-mean_iou: %.4f, pixel-mean_precision: %.4f, pixel-mean_recall: %.4f pixel-mean_auc: %.4f" \
                 % (mean_f1, mean_iou, mean_precision, mean_recall, mean_auc))

    return "Testing Finished!"

if __name__ == "__main__":

    if torch.cuda.is_available():
        args.device = 'cuda'
        cudnn.benchmark = False
        cudnn.deterministic = True
        args.num_gpu = torch.cuda.device_count()
    else:
        args.device = 'cpu'
    device = args.device
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    dataset_name = args.dataset
    
    # name the same snapshot defined in train script!
    args.exp = 'FU_' + dataset_name
    snapshot_path = "./models/{}/{}".format(args.exp, 'FU')
    snapshot_path = snapshot_path + '_' + args.split + '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(
        args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    if args.split == 'train':
        snapshot_path = snapshot_path + '_epo' + str(args.max_epochs)
    elif args.split == 'fine_tuned':
        snapshot_path = snapshot_path + '_epo' + str(args.ft_epochs)
    else:
        snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_' + str(args.optimizer)
    snapshot_path = snapshot_path + '_' + str(args.lr_name) + '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    print('snapshot_path:', snapshot_path)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') !=-1:#test baseline
        config_vit.patches.grid = (int(args.img_size/args.vit_patches_size), int(args.img_size/args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(device)
    net.to(device)
    if args.num_gpu > 1:
        net = nn.DataParallel(net)
    #print(net)
    snapshot = os.path.join(snapshot_path, 'epo{}.pth'.format(args.best_model))

    if not os.path.exists(snapshot): snapshot = snapshot_path+'/epo'+str(args.max_epochs-1)
    net.load_state_dict(torch.load(snapshot)['model'])
    #print('snapshot_model:',snapshot)
    print('load pretrained mode {} sucessfully.'.format(snapshot))

    snapshot_name = snapshot_path.split('/')[-1]
    log_folder = os.path.join('./test_log',args.exp, snapshot_name, args.test_data)
    args.test_log = log_folder
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_folder+'/'+'epo{}_results.txt'.format(args.best_model),
                        level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info(args.exp+'_'+snapshot_name)

    if args.is_save:
        args.visual_dir = './visualization'
        args.premask_path = os.path.join(args.visual_dir, args.exp, snapshot_name,args.test_data, 'epo{}_premask'.format(args.best_model))
        args.graph_path = os.path.join(args.visual_dir, args.exp, snapshot_name,args.test_data,'graphs')
        os.makedirs(args.premask_path, exist_ok=True)
        os.makedirs(args.graph_path, exist_ok=True)
        
    inference(args, model=net, split='test')
