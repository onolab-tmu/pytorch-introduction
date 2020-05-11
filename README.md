# pytorch-introduction

# スライド
- [Slide](https://gitpitch.com/onolab-tmu/pytorch-introduction)

# 準備
- Python 3系がインストール済み
    - Python 2系の利用はもうやめましょう
- pipやcondaなどのパッケージ管理ツールが利用可能
- (GPUを使う場合) CUDA, cuDNNがインストール済み

# このリポジトリのクローン
```
git clone https://github.com/popura/pytorch-introduction.git
```

# PyTorchのインストール
1. [Get Started](https://pytorch.org/get-started/locally/)から自分の環境を選択
1. Run this Command 欄に表示されたコマンドをターミナル上で実行
    - torch (PyTorch本体), torchvision (画像処理用パッケージ) がインストールされる
    - 仮想環境を作っている人は仮想環境を有効にしてから
1. torchaudio (音響信号処理用パッケージ) のインストール
    ```
    pip install torchaudio
    ```
1. その他パッケージのインストール
    ```
    pip install matplotlib
    ```
