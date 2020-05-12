import time
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torchsummary import summary
import trainer
import model
import transform as mytf
from dataset import AudioFolder, PreLoadAudioFolder

def parse_cmd_line_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-o', help='Output directory of models.',
                        type=str,
                        default='../model')
    parser.add_argument('--data_dir', help='Directory for storing datasets.',
                        type=str,
                        default='../data/small-acoustic-scenes')
    parser.add_argument('--batch_size', help='Batch size',
                        type=int,
                        default=4)
    parser.add_argument('--seed', help='Random seed',
                        type=int,
                        default=None)
    parser.add_argument('--epoch', help='Number of epochs',
                        type=int,
                        default=100)
    parser.add_argument('--lr', help='Initial learning rate',
                        type=float,
                        default=0.1)
    return parser.parse_args()

def print_cmd_line_arguments(args):
    print('-----Parameters-----')
    for key, item in args.__dict__.items():
        print(key, ': ', item)
    print('--------------------')

def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_data_loaders(path, batch_size):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)

    num_classes = 2
    transform = mytf.Compose([torchaudio.transforms.Resample(44100, 8000),
                              torchaudio.transforms.Spectrogram(n_fft=512)])

    # trainset = AudioFolder(p / 'train', transform=transform)
    trainset = PreLoadAudioFolder(p / 'train', transform=transform)
    # valset = AudioFolder(p / 'evaluate', transform=transform)
    valset = PreLoadAudioFolder(p / 'evaluate', transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return trainloader, valloader, num_classes

def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def show_history(train_accuracy, val_accuracy):
    plt.plot(range(len(train_accuracy)), train_accuracy,
             label='Accuracy for training data')
    plt.plot(range(len(val_accuracy)), val_accuracy,
             label='Accuracy for val data')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    args = parse_cmd_line_arguments()
    print_cmd_line_arguments(args)
    if args.seed is not None:
        set_random_seed(args.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainloader, valloader, num_classes = get_data_loaders(args.data_dir, args.batch_size)

    #net = model.ResNet('ResNet18', num_classes=num_classes)
    net = model.DCGANDiscriminator(num_classes=num_classes)
    net = net.to(device)
    summary(net, input_size=(2, 512, 512))

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=args.lr,
                              momentum=0.9, dampening=0,
                              weight_decay=0.0001, nesterov=False)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                               [args.epoch//2, 3*args.epoch//4],
                                               0.1)

    start_time = time.time()
    trainer = trainer.ClassifierTrainer(net, optimizer, criterion,
                                        trainloader, device)
    costs = []
    train_accuracy = []
    val_accuracy = []
    print('-----Training Started-----')
    for epoch in range(args.epoch):  # loop over the dataset multiple times
        loss = trainer.train()
        train_acc = trainer.eval(trainloader)
        val_acc = trainer.eval(valloader)
        print('Epoch: %03d/%03d | Loss: %.4f | Time: %.2f min | Acc: %.4f/%.4f'
              % (epoch+1, args.epoch, loss,
                 (time.time() - start_time)/60,
                 train_acc, val_acc))
        costs.append(loss)
        train_accuracy.append(train_acc)
        val_accuracy.append(val_acc)
        scheduler.step()

    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    print('-----Training Finished-----')

    save_model(net, args.o+'/trained_model.pth')

    show_history(train_accuracy, val_accuracy)