# pytorch-introduction

# スライド
- [Slide](https://gitpitch.com/onolab-tmu/pytorch-introduction)

# このリポジトリのクローン
```
git clone https://github.com/popura/pytorch-introduction.git
```

# 準備
## 前提
- (GPUを使う場合) CUDA, cuDNN がインストール済み

## Python 3系のインストール
- Python 2系の利用はもうやめましょう
- 通常インストール (Windows)
	1. [Python 公式](https://python.org) から 64bit 版 Python インストーラをダウンロード
	1. customize installation で、Add Python to environment variables にチェックをいれてインストール
	1. PowerShell で `python -V` が実行できることを確認

- Anaconda を使用したインストール (Windows)
	1. [Anaconda 公式](https://www.anaconda.com/download) から 64bit 版インストーラをダウンロード、インストール
	1. Anaoncda PowerShell Prompt で `python -V` が実行できることを確認

## 仮想環境の作成
- 以下，任意の仮想環境名を venv_name とする (例えば venv など，自分で決める)
- 通常環境
	1. 作成: pytorch-introduction フォルダ内で
		```
		% python -m venv venv_name
		```
	1. 有効化:
		```
		% ./venv_name/Scripts/activate
		```
	- Windows 環境でエラーを吐く場合
		1. 管理者権限で PowerShell を実行
		1. `% Get-ExecutionPolicy` を入力すると Restricted が帰ってくることを確認
		1. `% Set-EcecutionPolicy AllSigned` を入力、`Y` を入力
		1. pytorch-introduction フォルダ内で `./venv/Scripts/activate`

- Anaconda 環境
	1. 作成: 任意の場所で
		```
		% conda create -n venv_name
		% y
		```
	1. 有効化:
		```
		% conda activate venv_name
		```

## PyTorchのインストール
- 一般
	1. [Get Started](https://pytorch.org/get-started/locally/)から自分の環境を選択
	1. Run this Command 欄に表示されたコマンドをターミナル上で実行
		- torch (PyTorch本体), torchvision (画像処理用パッケージ) がインストールされる
		- 仮想環境を作っている人は仮想環境を有効にしてから

- 通常環境 + GPUなし (2020.05.13 時点)
	```
	% pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
	```

- Anaconda 環境 + GPUなし
	```
	% conda install pytorch torchvision cpuonly -c pytorch
	```


## torchaudio (音響信号処理用パッケージ) のインストール
- windows 環境以外は `pip install torchaudio` など
- windows 用ビルドはないようなので自分でビルドする必要がある
1. pytorch-introduction 以外の場所で，[torchaudio の github リポジトリ](https://github.com/pytorch/audio) を clone  (audioフォルダが作成される)
	```
	% git clone https://github.com/pytorch/audio
	```
1. pytorch-introduction の仮想環境を有効にした状態で，audio内に移動し、`python setup.py install` を入力

## その他必要なパッケージのインストール
- 通常環境
	```
	% pip install matplotlib torchsummary soundfile
	```

- Anaconda 環境
	```
	% conda install matplotlib
	% pip install torchsummary soundfile
	```
	- 通常，conda のみを用いてインストールするべきだが，torchsummary は pip でしかインストールできない
	- conda-forge で配布されている soundfile は最新の Python に対応していない?
