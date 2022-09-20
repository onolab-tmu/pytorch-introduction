import torch
import torch.nn as nn
import torch.optim as optim

from model import SimpleDNN
from dataset import AcousticSceneDataset
from trainer import ClassifierTrainer


def get_data_loaders():
    trainset = AcousticSceneDataset(
        root="./data/small-acoustic-scenes",
        mode="train",
        transforms=
    )
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=16,
        shuffle=True,
        num_workers=4
    )

    # Although the training dataset is also used for validation here,
    # a validation dataset is generally different from a training dataset.
    valset = AcousticSceneDataset(
        root="./data/small-acoustic-scenes",
        mode="train",
        transforms=
    )
    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    return trainloader, valloader

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 学習・検証データセットを読み込むクラス（DataLoaderクラス）を作る
    trainloader, valloader = get_data_loaders()

    # DNNを作る
    net = SimpleDNN()

    # 損失関数やその他ハイパーパラメータの定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(lr=0.001)
    schedular = optim.lr_scheduler.MultiStepLR(optimizer, 0.1, [50, 100])

    # 学習
    trainer = ClassifierTrainer(
        net,
        optimizer,
        criterion,
        trainloader,
        scheduler=schedular,
        device=device
    )
    trainer.train(100, valloader)

    # モデル保存
    output_dir = Path("./result")
    if not output_dir.exists()
        output_dir.mkdir(parents=True)
    torch.save(net.state_dict(), output_dir / "trained_weights.ph")


if __name__ == "__main__":
    main()