import argparse





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', default='../data1 train_data.pth', type=str,
                    help='path of train data')
    parser.add_argument('--batch_size', default=20, type=int,
                        help='size of batches')
    parser.add_argument('--ge_bin_size', default=30, type=int,
                        help='window size used in GAN')
    parser.add_argument('--latent_dim', default=50, type=int,
                        help='size of random vector')
    parser.add_argument('--label_dim', default=100, type=int,
                        help='size of embedded label')
    parser.add_argument('--seq_len', default=1200, type=int,
                        help='length of temporal dimension')
    parser.add_argument('--g_lr', default=0.01, type=float,
                        help='learning rate of generator')
    parser.add_argument('--d_lr', default=0.0001, type=float,
                        help='learning rate of discriminator')

    parser.add_argument(
        '--epoch',
        type=int,
        default=1000,
        help='number of epochs of training')
    parser.add_argument(
        '--channels',
        type=int,
        default=90,
        help='length of spatial dimension')
    parser.add_argument(
        '--depth',
        type=int,
        default=1,
        help='number of transformer blocks')
    parser.add_argument(
        '--attn_drop_rate',
        type=float,
        default=0.5,
        help='dropout rate of attention module')
    parser.add_argument(
        '--forward_drop_rate',
        type=float,
        default=0.5,
        help='dropout rate of feedforward module')
    parser.add_argument(
        '--time_seg',
        type=tuple,
        default=(5,4),
        help='segments of temporal segmentation')
    parser.add_argument(
        '--num_heads',
        type=int,
        default=5,
        help='number of attention heads')

    parser.add_argument(
        '--output_path',
        type=str,
        default='../results/generated_data.pth',
        help='path of generated data')
    
    opt = parser.parse_args()

    return opt