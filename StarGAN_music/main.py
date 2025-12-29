import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn


def str2bool(v):
    return v.lower() in ('true')

def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    score_loader = get_loader(config.score_dir, config.encoding, config.attr_path, config.selected_attrs,
                              config.batch_size,'MidiCaps', config.mode, config.num_workers)
    

    # Solver for training and testing StarGAN.
    solver = Solver(score_loader, config)
    
    if config.mode == 'train':
        if config.dataset in ['MidiCaps']:
            solver.train()
        elif config.dataset in ['Both']:
            solver.train_multi()
    elif config.mode == 'test':
        if config.dataset in ['MidiCaps']:
            solver.test()
        elif config.dataset in ['Both']:
            solver.test_multi()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=5, help='dimension of domain labels (1st dataset)')
    parser.add_argument('--c2_dim', type=int, default=8, help='dimension of domain labels (2nd dataset)')
    parser.add_argument('--celeba_crop_size', type=int, default=178, help='crop size for the CelebA dataset')
    parser.add_argument('--rafd_crop_size', type=int, default=256, help='crop size for the RaFD dataset')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--g_conv_dim', type=int, default=64, help='number of conv filters in the first layer of G')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--g_repeat_num', type=int, default=6, help='number of residual blocks in G')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--lambda_cls', type=float, default=1, help='weight for domain classification loss')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for reconstruction loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    
    # Generator configuration.
    parser.add_argument('--g_modelpath', type=str, default="../Amadeus/models/Amadeus-S", help='path to the generator model')
    parser.add_argument('--generate_length', type=int, default=100, help='length of the generated sequence')
    parser.add_argument('--sampling_method', type=str, choices=('top_p', 'top_k'), default="top_k", help='sampling method for generation')
    parser.add_argument('--threshold', type=float, default=0.99, help='threshold for sampling method')
    parser.add_argument('--temperature', type=float, default=1.15, help='temperature for sampling method')
    
    # Discriminator configuration.
    parser.add_argument('--d_modelpath', type=str, default="../Moonbeam-MIDI-Foundation-Model/models/emotion_classification-v1", help='path to the discriminator model')
    
    # Training configuration.
    parser.add_argument('--dataset', type=str, default='MidiCaps', choices=['MidiCaps', 'Both'])
    parser.add_argument('--batch_size', type=int, default=1, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each G update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    parser.add_argument('--selected_attrs', '--list', nargs='+', help='selected attributes for Music dataset',
                        default=['funk', 'celtic', 'instrumentalpop', 'ambient', 'reggae', 'popfolk', 'dance', 'rock', 'classical', 'instrumentalrock', 'folk', 'poprock', 'indie', 'hiphop', 'blues', 'experimental', 'punkrock', 'jazz', 'electronic', 'techno', 'jazzfusion', 'pop', 'alternative', 'electropop', 'soundtrack', 'trance', 'house', 'metal', 'world', 'symphonic', 'lounge', 'easylistening', 'orchestral', 'country', 'newage', 'latin', 'drumnbass', '80s', '90s', 'swing', 'chillout', 'synthpop', 'movie', 'christmas', 'heavy', 'corporate', 'action', 'romantic', 'energetic', 'background', 'children', 'calm', 'adventure', 'motivational', 'summer', 'funny', 'dramatic', 'cool', 'positive', 'emotional', 'holiday', 'deep', 'love', 'dark', 'dream', 'advertising', 'happy', 'soundscape', 'film', 'melodic', 'drama', 'uplifting', 'epic', 'ballad', 'sad', 'relaxing', 'party', 'trailer', 'inspiring', 'soft', 'slow', 'game', 'retro', 'fun', 'meditative', 'sport', 'space', 'commercial', 'documentary', 'upbeat', 'Eb major', 'B major', 'Bb major', 'F# minor', 'F# major', 'G# minor', 'A major', 'B minor', 'E minor', 'D minor', 'F minor', 'G minor', 'F major', 'Eb minor', 'C major', 'A minor', 'G major', 'D major', 'C# major', 'Bb minor', 'Ab major', 'C# minor', 'C minor', 'E major'])

    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=True)

    # Directories.
    parser.add_argument('--score_dir', type=str, default='../Amadeus/dataset/MidiCaps/corpus/tuneidx_')
    parser.add_argument('--encoding', type=str, default='nb8')  
    parser.add_argument('--attr_path', type=str, default='../Dataset/MidiCaps/train.json')
    parser.add_argument('--vocab_path', type=str, default='../Amadeus/models/Amadeus-S/files/checkpoints/vocab_LakhALLFined_nb8.json')
    model_name = "MidiCaps-v0"
    parser.add_argument('--log_dir', type=str, default=f'result/{model_name}/logs')
    parser.add_argument('--model_save_dir', type=str, default=f'result/{model_name}/models')
    parser.add_argument('--sample_dir', type=str, default=f'result/{model_name}/samples')
    parser.add_argument('--result_dir', type=str, default=f'result/{model_name}/results')

    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)