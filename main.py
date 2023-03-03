import cfg
import argparse
import vision.train as vision


parser = argparse.ArgumentParser(description='Train or evaluate a pointnet autoencoder')
parser.add_argument('mode', choices=['train', 'eval', 'traineval'])
parser.add_argument('model', choices=['GTPredictor'])
parser.add_argument('dataset', default='Lift')
# parser.add_argument('--output', default='vision/output',
#                     help='path to output data (for evaluation)')
parser.add_argument('--batch_size', default=cfg.vision_batch_size, type=int,
                    help='batch size for training')
parser.add_argument('--num_epochs', default=cfg.vision_epochs, type=int,
                    help='number of epochs to train for')
a = parser.parse_args()


print(f'device = {cfg.device}')

if a.mode in ['train', 'traineval']:
    vision.train(a.model, a.dataset, a.num_epochs, a.batch_size)
# if a.mode in ['eval', 'traineval']:
#     vision.eval(a.model, a.valset, a.output, a.batch_size)

