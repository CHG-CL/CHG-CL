import time
import argparse
import pickle
from model import *
from utils import *
import warnings

warnings.filterwarnings("ignore")


def init_seed(seed=None):
    if seed is None:
        seed = 2024
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='30music', help='diginetica/30music/Tmall/')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=30)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--phi', type=float, default=2.0)
parser.add_argument('--temp', type=float, default=0.1, help='The temperature of the SSL mechanism.')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--decay_count', type=int, default=0,
                    help='Once it equals to the decay num, the lr_dc would proceed.')
parser.add_argument('--decay_num', type=int, default=3, help='The epoch needed lr_dc.')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3,
                    help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=1, help='GNN layer num')
parser.add_argument('--dropout_attribute', type=float, default=0.5, help='Dropout rate of the mirror graph generation.')
parser.add_argument('--dropout_score', type=float, default=0.2,
                    help='Dropout rate of the session representation generation.')
parser.add_argument('--validation', default=False, action='store_true', help='when you need a validation dataset.')
parser.add_argument('--sample_num', type=int, default=50, help='the sample num of attribute-same items.')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument("--max_seq_length", default=70, type=int)
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--k', type=int, default=10, help='The numbers of Neighbor for session node')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--atten_dropout', type=float, default=0.2)
parser.add_argument('--num_attention_heads', type=int, default=10)
parser.add_argument('--sessionGAT_heads', type=int, default=1)
parser.add_argument('--model_path', type=str, default='./best_model', help='if you need to store your model.')
parser.add_argument('--aggregation', default='sum',
                    help='The aggregation operations of session representation'
                         ' and session context information')
parser.add_argument('--gmlp_layers', type=int, default=1)
parser.add_argument('--gmlp_dropout', type=float, default=0.5)
parser.add_argument('--layer_norm_eps', type=float, default=1e-8)
opt = parser.parse_args()


def main():
    init_seed(2024)

    if opt.dataset == 'diginetica':
        num_node = 43098
        opt.n_iter = 3
        opt.attribute_kinds = 1
        opt.phi = 2.5
        opt.temp = 0.1
        opt.sample_num = 40
        opt.k = 10
        opt.num_attention_heads = 10
        opt.atten_dropout = 0.2
    elif opt.dataset == '30music':
        num_node = 132648
        opt.n_iter = 2
        # 0.8--->0.5(best)-->0.4(40.47)--->0.3（40.396）----0.35(40.25)---->0.45
        opt.dropout = 0.5
        opt.attribute_kinds = 1
        opt.phi = 1.5
        # 0.05-->0.1-->0.01
        opt.temp = 0.05
        # 10--->5--->15--->12(40.09)--->8(40.1)
        opt.sample_num = 9
        # 15-->10
        opt.k = 10
        # 0.2(best)-->0.5(40.25)
        opt.atten_dropout = 0.2
        opt.num_attention_heads = 10
    elif opt.dataset == 'Tmall':
        num_node = 40728
        opt.hiddenSize = 210
        opt.n_iter = 2
        opt.dropout_attribute = 0.5
        opt.attribute_kinds = 2
        opt.phi = 0.9
        opt.temp = 0.05
        opt.sample_num = 5
        opt.dropout = 0.6

        opt.num_attention_heads = 2
        opt.sessionGAT_heads = 1
        opt.atten_dropout = 0.2

        opt.k = 20
    else:
        num_node = 310

    train_data = pickle.load(open('datasets/' + opt.dataset + '/train.txt', 'rb'))
    product_attributes = load_json('datasets/' + opt.dataset + '/product_attributes.json')
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('datasets/' + opt.dataset + '/test.txt', 'rb'))

    train_data = Data(train_data, product_attributes, opt, opt.max_seq_length)
    test_data = Data(test_data, product_attributes, opt, opt.max_seq_length)

    model = CombineGraph(opt, num_node)
    model = trans_to_cuda(model)

    print(opt)
    start = time.time()
    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0

    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        train_data.sample_num = opt.sample_num
        [hit_10, mrr_10], [hit_20, mrr_20] = train_test(model, opt, train_data, test_data)
        flag = 0
        if hit_20 >= best_result[0]:
            # torch.save(model.state_dict(), f"{opt.model_path}/{opt.dataset}/p10-{hit_10:.4f}_mrr10-{mrr_10:.4f}.pth")
            best_result[0] = hit_20
            best_epoch[0] = epoch
            flag = 1
        if mrr_20 >= best_result[1]:
            # torch.save(model.state_dict(), f"{opt.model_path}/{opt.dataset}/p10-{hit_10:.4f}_mrr10-{mrr_10:.4f}.pth")
            best_result[1] = mrr_20
            best_epoch[1] = epoch
            flag = 1
        print('Current Result:')
        print('\tPrecision@10:\t%.4f\tMRR@10:\t%.4f\tPrecision@20:\t%.4f\tMRR@20:\t%.4f' % (
            hit_10, mrr_10, hit_20, mrr_20))
        print('Best Result:')
        print('\tPrecision@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
            best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
