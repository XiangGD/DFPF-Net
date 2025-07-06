import argparse
import os
import random
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from trainer import trainer_forensic
from networks.vit_seg_modeling import ForensicTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from networks.mvssnet import get_mvss

parser = argparse.ArgumentParser()
#dataset parameters
parser.add_argument('--root_path', type=str,default='../datasets', help='root dir for data')
parser.add_argument('--list_dir', type=str,default='./lists', help='list dir')
parser.add_argument('--num_classes', type=int, default=1, help='output channel of network')
parser.add_argument('--train_data', type=str, default=['IML-MUST'], #default=['CASIA1','CASIA2','Coverage','NIST2016','Columbia','DEFACTO'],\
                    help='training dataset of network')
parser.add_argument('--ft_data', type=list, default=['CASIA2','Coverage','NIST2016','Columbia','IMD2020'],
                    help='fine tune dataset of network')
parser.add_argument('--val_data', type=str, default='Coverage', help='valuation dataset of network')
parser.add_argument('--test_data', type=str, default='Coverage', help='testing dataset of network')
#model parameters
parser.add_argument('--vit_name', type=str,default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,default=16, help='vit_patches_size, default is 16')
parser.add_argument('--train_epochs', type=int,default=120, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=36, help='batch_size per gpu')
parser.add_argument('--num_gpu', type=int, default=0, help='total gpu number, if only cup, num_gpu equal 0')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--n_skip', type=int, default=3, help='using number of skip-connect, default is num')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
parser.add_argument('--img_size', type=int, default=256, help='input patch size of network input')
parser.add_argument('--lr_name', type=str, default='Customized', help='lr schedule type')
parser.add_argument('--optimizer', type=str, default='Adam', help='optimizier type，SGD or Adam')
parser.add_argument('--base_lr', type=float, default=0.00012,help='segmentation network learning rate')
parser.add_argument('--save_interval', type=int, default=5, help='frequency of perform checkpoint')
#fine_tuned parameters
parser.add_argument('--split', type=str,default='train',help='split train model or fine tune model')
parser.add_argument('--freeze', type=bool,default=False,help='Froze some layers to fine-tuned the model')
parser.add_argument('--ft_epochs', type=int,default=120, help='max epochs for fine-tuned the model')
parser.add_argument('--train_best_model', type=int, default=88, help='best pre_train weights')
parser.add_argument('--ft_load_path', type=str, default='', help='fine tuned init weights load path')
# checkpiont parameters
parser.add_argument('--resume', type=bool, default=False, help='resume from checkpoint')
parser.add_argument('--continue_model', type=int,default=94, help='resume from checkpoint')
parser.add_argument('--ckpt_path', type=str, default='', help='resume path')
#evaluation parameters
parser.add_argument('--premask_save_path', type=str, default=None, help='save path of precision masks of evaluation')

args = parser.parse_args()

if 'LOCAL_RANK' not in os.environ:
    os.environ['LOCAL_RANK'] = str(args.local_rank)

if __name__ == "__main__":

    if torch.cuda.is_available():
        args.device = 'cuda'
        cudnn.benchmark = False
        cudnn.deterministic = True
        #args.num_gpu = torch.cuda.device_count()
        args.num_gpu = 1 #caculate the training time
    else:
        args.device = 'cpu'

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.train_data
    args.is_pretrain = True
    args.exp = 'FU_' + dataset_name
    
    snapshot_path = "./models/{}/{}".format(args.exp, 'FU')
    snapshot_path = snapshot_path + '_' + args.split + '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.train_epochs) if args.split == 'train' else snapshot_path + '_epo' + str(args.ft_epochs)
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_' + str(args.optimizer)
    snapshot_path = snapshot_path + '_' + str(args.lr_name)+ '_lr' + str(args.base_lr)
    snapshot_path = snapshot_path + '_'+ str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    #print('snapshot_path:',snapshot_path)
    
    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    args.pretrain_path = config_vit.pretrained_path
    
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
    net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).to(args.device)
    mvss_net = get_mvss().to(args.device)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    assert args.split == 'train' or args.split == 'fine_tuned', 'split not train or ft !'
    if args.split == 'train':
        net.load_from(weights=np.load(config_vit.pretrained_path))
        print('load pretrained_dict successfully!')
    else:
        ft_load_path = snapshot_path.replace('_epo'+str(args.ft_epochs), '_epo'+str(args.train_epochs))
        ft_load_path = ft_load_path.replace('_'+args.split, '_train')
        args.ft_load_path = os.path.join(ft_load_path, 'epo{}.pth'.format(args.train_best_model))
        print( args.ft_load_path)
        assert os.access(args.ft_load_path, os.F_OK), 'fine tuned load path not exits!'

    if args.resume:
        args.ckpt_path = os.path.join(snapshot_path,'epo{}.pth'.format(args.continue_model))
        assert os.access(args.ckpt_path, os.F_OK), 'resume path not exits!'

    trainer = {'IML-MUST':  trainer_forensic,
               'CASIA2': trainer_forensic,
               }
    trainer[dataset_name](args, net, snapshot_path)
    
