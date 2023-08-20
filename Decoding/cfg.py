import argparse





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='../data1 train_data.pth', type=str,
                    help='path of train data')
    parser.add_argument('--test_path', default='../data1 test_data.pth', type=str,
                    help='path of test data')
    parser.add_argument('--generated_path', default=None, type=str,
                    help='path of generated data')
    parser.add_argument('--batch_size', default=20, type=int,
                        help='size of batches')
    parser.add_argument('--bin_size', default=60, type=int,
                        help='window size used in decoding models')
    parser.add_argument('--step_size', default=100, type=int,
                        help='step size of scheduler')
    parser.add_argument('--gamma', default=0.1, type=float,
                        help='gamma of scheduler')
    parser.add_argument('--decay', default=1e-3, type=float,
                        help='weight decay in Adam')
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='learning rate of generator')

    parser.add_argument(
        '--epoch',
        type=int,
        default=300,
        help='number of epochs of training')
    parser.add_argument(
        '--model_name',
        type=str,
        default='ReSNN',
        help='name of decoding model',
        choices=['ReSNN','LSTM','FCSNN'])
    
    
    opt = parser.parse_args()

    return opt