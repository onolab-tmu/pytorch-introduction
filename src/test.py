import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import trainer
import model
import transform as mytf
from dataset import VCTKWrapper, LIBRISPEECHWrapper, SPEECHCOMMANDSWrapper, AudioFolder


def parse_cmd_line_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model', help='Model path',
                        type=str,
                        default='../model')
    parser.add_argument('--data_dir', help='Directory for storing datasets.',
                        type=str,
                        default='../data/small-acoustic-scenes')
    parser.add_argument('--batch_size', help='Batch size',
                        type=int,
                        default=1)
    return parser.parse_args()


def print_cmd_line_arguments(args):
    print('-----Parameters-----')
    for key, item in args.__dict__.items():
        print(key, ': ', item)
    print('--------------------')


def get_data_loaders(path, batch_size):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    classes = ('car', 'home')
    num_classes = len(classes)
    transform = mytf.Compose([torchaudio.transforms.Resample(44100, 8000),
                              torchaudio.transforms.Spectrogram(n_fft=512)])

    testset = AudioFolder(p / 'evaluate', transform=transform)

    loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return loader, num_classes, classes


def examples(loader, net, classes):
    dataiter = iter(loader)
    waveforms, labels = dataiter.next()
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[i]] for i in range(len(labels))))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[i]] for i in range(len(labels))))


def calculate_accuracy(loader, net, num_classes, classes):
    class_correct = list(0. for i in range(num_classes))
    class_total = list(0. for i in range(num_classes))
    with torch.no_grad():
        for data in loader:
            waveforms, labels = data
            outputs = net(waveforms)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))



if __name__ == "__main__":
    args = parse_cmd_line_arguments()
    print_cmd_line_arguments(args)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    loader, num_classes, classes = get_data_loaders(args.data_dir, args.batch_size)

    net = model.DCGANDiscriminator()
    net.load_state_dict(torch.load(args.model))
    net.eval()

    examples(loader, net, classes)
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
