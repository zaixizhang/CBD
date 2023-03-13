import argparse

def get_arguments():
    parser = argparse.ArgumentParser()

    # various path
    parser.add_argument('--weight_root', type=str, default='./weight', help='save model weights')
    parser.add_argument('--log_root', type=str, default='./logs', help='logs are saved here')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--model_name', type=str, default='WRN-16-1', help='name of model')
    parser.add_argument('--load_fixed_data', type=int, default=0, help='load the local poisoned dataest')
    parser.add_argument('--checkpoint_root', type=str, default='./weight/backdoored_model/', help='path of backdoored model')
    parser.add_argument('--data_root', type=str,default='./data/',help='path of data')

    # training hyper parameters
    parser.add_argument('--print_freq', type=int, default=200, help='frequency of showing training results on console')
    parser.add_argument('--tuning_epochs', type=int, default=100, help='number of tune epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes')

    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--disentangle', type=int, default=0)
    parser.add_argument('--interval', type=int, default=5, help='frequency of save model')

    # others
    parser.add_argument('--seed', type=int, default=2, help='random seed')
    parser.add_argument('--note', type=str, default='try', help='note for this run')

    # backdoor attacks
    parser.add_argument('--inject_portion', type=float, default=0.1, help='ratio of backdoor samples')
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')
    parser.add_argument("--input_height", type=int, default=32)
    parser.add_argument("--input_width", type=int, default=32)

    # for dynamic
    '''
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--temps", type=str, default="./temps")

    parser.add_argument("--input_height", type=int, default=32)
    parser.add_argument("--input_width", type=int, default=32)
    parser.add_argument("--input_channel", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--batchsize", type=int, default=128)
    parser.add_argument("--lr_G", type=float, default=1e-2)
    parser.add_argument("--lr_C", type=float, default=1e-2)
    parser.add_argument("--lr_M", type=float, default=1e-2)
    parser.add_argument("--schedulerG_milestones", type=list, default=[200, 300, 400, 500])
    parser.add_argument("--schedulerC_milestones", type=list, default=[100, 200, 300, 400])
    parser.add_argument("--schedulerM_milestones", type=list, default=[10, 20])
    parser.add_argument("--schedulerG_lambda", type=float, default=0.1)
    parser.add_argument("--schedulerC_lambda", type=float, default=0.1)
    parser.add_argument("--schedulerM_lambda", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=600)
    parser.add_argument("--lambda_div", type=float, default=1)
    parser.add_argument("--lambda_norm", type=float, default=100)
    parser.add_argument("--num_workers", type=float, default=4)

    parser.add_argument("--attack_mode", type=str, default="all2one", help="all2one or all2all")
    parser.add_argument("--p_attack", type=float, default=0.1)
    parser.add_argument("--p_cross", type=float, default=0.1)
    parser.add_argument("--mask_density", type=float, default=0.032)
    parser.add_argument("--EPSILON", type=float, default=1e-7)

    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)
    
    #for wanet
    parser.add_argument("--checkpoints", type=str, default="./checkpoints")
    parser.add_argument("--temps", type=str, default="./temps")
    parser.add_argument("--continue_training", action="store_true")

    parser.add_argument("--attack_mode", type=str, default="all2one")

    parser.add_argument("--input_channel", type=int, default=3)
    parser.add_argument("--num_classes", type=int, default=43)

    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--lr_C", type=float, default=1e-2)
    parser.add_argument("--schedulerC_milestones", type=list, default=[100, 200, 300, 400])
    parser.add_argument("--schedulerC_lambda", type=float, default=0.1)
    parser.add_argument("--n_iters", type=int, default=1000)
    parser.add_argument("--num_workers", type=float, default=6)

    parser.add_argument("--pc", type=float, default=0.1)
    parser.add_argument("--cross_ratio", type=float, default=2)  # rho_a = pc, rho_n = pc * cross_ratio

    parser.add_argument("--random_rotation", type=int, default=10)
    parser.add_argument("--random_crop", type=int, default=5)

    parser.add_argument("--s", type=float, default=0.5)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument(
        "--grid-rescale", type=float, default=1
    )
    '''
    return parser
