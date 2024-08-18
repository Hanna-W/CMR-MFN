import argparse

parser = argparse.ArgumentParser(description="PyTorch implementation of Confusion Mixup Regularized Multimodal Fusion Network(CMR-MFN)")

parser.add_argument('dataset', type=str, choices=['UESTC-MMEA-CL'])
parser.add_argument('modality', type=str, nargs='+', choices=['RGB', 'Gyrospec', 'Accespec'],
	                default=['RGB', 'Accespec', 'Gyrospec'])
parser.add_argument('--train_list', type=str)
parser.add_argument('--val_list', type=str)
parser.add_argument('--mpu_path', type=str, default=None)

# ========================= CL Configs =============================
parser.add_argument('--config', type=str,
                    help='json file of CL algorthms config')

# ========================= Model Configs ==========================
parser.add_argument('--arch', type=str, default="ViT")
parser.add_argument('--num_segments', type=int, default=3)
parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--fusion_type', choices=['concat', 'attention'],
                    default='attention')


# ========================= Learning Configs ==========================
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[10, 20], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--clip-gradient', '--gd', default=None, type=float,
                    metavar='W', help='gradient norm clipping (default: disabled)')
parser.add_argument('--freeze', '-f', action='store_true',
                    help='freeze all weights except fusion and classification')


# ========================= Runtime Configs ==========================
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--device', type=list, default=["5"])


# ============================ Others ===============================
parser.add_argument('--experiment_suffix', default="", type=str)