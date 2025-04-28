import argparse
import os

## path
threedpw_dataset_path = "./data/3dpw/"
somof_dataset_path = "./data/somof_data_3dpw/"
amass_dataset_path = "./data/amass/"


## data
input_length = 16
output_length = 14
num_joint = 13
num_person = 2


# training
batch_size = 256
learning_rate = 0.001
num_epoch = 512

device = 'cuda'


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', 
                        type=str, 
                        default="3dpw",
                        help='dataset name')

    ## path setting
    parser.add_argument('--log_dir', 
                        type=str, 
                        default=os.path.join(os.getcwd(), 'logs/'),
                        help='dir for saving logs')
    
    parser.add_argument('--somof_dataset_path', 
                        type=str, 
                        default=somof_dataset_path,
                        help='path of somof dataset directory')
    
    parser.add_argument('--amass_dataset_path', 
                        type=str, 
                        default=amass_dataset_path,
                        help='path of amass dataset directory')
    
    parser.add_argument('--threedpw_dataset_path', 
                        type=str, 
                        default=threedpw_dataset_path,
                        help='path of 3dpw dataset directory')
    
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=batch_size,
                        help='batch size to train')
    parser.add_argument('--lr', 
                        type=float, 
                        default=learning_rate,
                        help='initial learing rate')
    parser.add_argument('--num_epoch', 
                        type=int, 
                        default=num_epoch,
                        help='#epochs to train')
    
    parser.add_argument('--device', 
                        type=str, 
                        default=device,
                        help='set device for training')
    parser.add_argument('--ckp',
                        type=str,
                        default='',
                        help='path of the model')
    
    parser.add_argument('--input_len', 
                        type=int, 
                        default=input_length,
                        help='input sequence length')
    parser.add_argument('--output_len', 
                        type=int, 
                        default=output_length,
                        help='output sequence length')
    parser.add_argument('--J', 
                        type=int, 
                        default=num_joint,
                        help='number of joints')  
    parser.add_argument('--N', 
                        type=int, 
                        default=num_person,
                        help='number of person in a scene')    
    
    parser.add_argument('--finetune',
                        type=bool,
                        default=False,
                        help='whether to finetune (or pretrain when False)')

    parser.add_argument('--ablation_exclude', 
                        nargs='+',
                        help='list of options to exclude for ablation study')    

    args = parser.parse_args()
    
    args.log_dir = os.path.join(args.log_dir, args.dataset+"/")
        
    if args.ablation_exclude is None:
        args.ablation_exclude = []
    
    print("Logging into dir:", args.log_dir)

    return args
