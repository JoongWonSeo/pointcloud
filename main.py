import argparse
import vision.train as vision


parser = argparse.ArgumentParser(description='Train or evaluate a pointnet autoencoder')
parser.add_argument('mode', choices=[
                    'train', 'eval', 'traineval'], help='train or evaluate the model')
parser.add_argument('--model', default='vision/weights/PC_AE.pth',
                    help='path to model weights (to save during training or load during evaluation)')
parser.add_argument('--input', default='vision/input',
                    help='path to training data (for training) or input data (for evaluation)')
parser.add_argument('--output', default='vision/output',
                    help='path to output data (for evaluation)')
parser.add_argument('--device', default='cuda:0',
                    help='device to use for training or evaluation')
parser.add_argument('--batch_size', default=25, type=int,
                    help='batch size for training')
parser.add_argument('--num_epochs', default=100, type=int,
                    help='number of epochs to train for')
parser.add_argument('--eps', default=0.002, type=float,
                    help='epsilon for EMD')
parser.add_argument('--iterations', default=5000, type=int,
                    help='number of iterations for EMD')
a = parser.parse_args()


print(f'device = {a.device}')

if a.mode == 'train':
    vision.train(a.input, a.model, a.num_epochs,
          a.batch_size, a.eps, a.iterations, a.device)
elif a.mode == 'eval':
    vision.eval(a.model, a.input, a.output, a.eps, a.iterations, a.device)
elif a.mode == 'traineval':
    vision.train(a.input, a.model, a.num_epochs,
          a.batch_size, a.eps, a.iterations, a.device)
    vision.eval(a.model, a.input, a.output, a.eps, a.iterations, a.device)
