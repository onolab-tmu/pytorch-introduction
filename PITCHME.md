### PyTorch勉強会
[GitHub(https://github.com/onolab-tmu/pytorch-introduction)](https://github.com/onolab-tmu/pytorch-introduction)  
[スライド(https://gitpitch.com/onolab-tmu/pytorch-introduction)](https://gitpitch.com/onolab-tmu/pytorch-introduction)  
Yuma Kinoshita

+++

### はじめに
#### もくじ
- この勉強会について
- Gitのインストール
- Python3のインストール
- Pythonの仮想環境作成（任意）
- PyTorchのインストール

+++

### この勉強会について
<img src="https://github.com/pytorch/pytorch/blob/master/docs/source/_static/img/pytorch-logo-dark.png" alt="pytorch logo" title="pytorch logo" width="200">
- 深層学習フレームワークPyTorchに触れる
  - 現在はPyTorchかTensorflow2の二択
  - 画像処理分野ではPyTorch実装が多く存在

+++

### 準備：Git
<img src="https://git-scm.com/images/logos/downloads/Git-Logo-2Color.png" alt="git logo" title="git logo" width="200">

#### Gitとは？
- [wikipedia](https://ja.wikipedia.org/wiki/Git)によれば…
  > プログラムのソースコードなどの変更履歴を
    記録・追跡するための分散型バージョン管理システム
- 変更をタイムマシンのように巻き戻せる
- 複数人での共有が簡単

+++

### Gitのインストール
- Windows
  - [Git for Windows](https://gitforwindows.org/)
- Mac
  ```
  $ brew install git
  ```
- Ubuntu
  ```
  $ sudo apt install git
  ```

- インストールできたことを確認
  ```
  $ git --version
  ```
  - Windowsの場合はGit Bashを起動

+++

### GitHubに登録
<img src="https://github.githubassets.com/images/modules/logos_page/Octocat.png" alt="github logo" title="github logo" width="200">

#### Githubとは？
- 無料で使えるGitサーバ
- [ここ](https://github.com/)からアカウントを作成
  - 全世界に公開されるので個人情報には注意

- アカウントができたら
  [Ono Lab @ TMU](https://github.com/onolab-tmu/)に参加

+++

### リポジトリをクローンする
#### リポジトリとは？
- バージョン管理を行う基本単位
- onolab-tmu/pytorch-introductionというリポジトリで，
  この勉強会用のファイルのバージョン管理を行っている

+++

### リポジトリのクローン
- Git bash，または端末を開く
- 勉強会のファイルを置きたいディレクトリに移動
  ```
  $ cd ファイルを置きたいディレクトリ
  ```
- githubからリポジトリをクローン
  ```
  $ git clone https://github.com/onolab-tmu/pytorch-introduction
  ```

+++

### Python3のインストール
- Windows
  - [公式ページ](https://www.python.org/)
    のインストーラーを実行
  - C:\\usr\\local\\bin\\Python3x と  
    C:\\usr\\local\\bin\\Python3x\\Scripts をPATHに追加
- Mac
  ```
  $ brew install python3
  ```
- Ubuntu
  ```
  $ sudo apt install python3
  ```

+++

### 仮想環境の構築
- Python開発ではプロジェクトごとに  
  仮想環境を作ることが一般的
  - パッケージの管理が簡単
  - 共有のサーバで人の環境を壊す or 壊される心配がない
- Python 3.4以降であれば，
  仮想環境を作成する*virtualenv*が公式に含まれているのでそれを使う

+++

### 仮想環境の構築
1. プロジェクトのディレクトリへ移動  
  （gitからcloneしたpytorch-introductionの下）
  ```
  $ cd hogehoge/pytorch-introduction/
  ```
1. virtualenvを使ってpython 3.6の環境を作る
  ```
  $ python3 -m venv venv
  ```
  venvというフォルダが作成されれば成功

+++

### 仮想環境の有効化・無効化
- 有効化
  - Windows
    ```
    $ .\venv\Scripts\activate
    ```
  - Mac, Ubuntu
    ```
    $ source venv/bin/activate
    ```
  - 仮想環境が有効化されると，  
    ターミナル左に(venv)と表示される
- 無効化
  ```
  $ deactivate
  ```

+++

### 開発環境を整える
#### 開発環境って？
- プログラムはテキストファイル（.txtと一緒）
- テキストエディタがあれば編集できる
- **ただし…** シンプルなテキストエディタで  
  プログラムを書くのは苦行

  <img src="https://raw.githubusercontent.com/tmu-sig/learn-python/master/fig/texteditor.png" alt="text editor logo" title="text editor logo" width="200">

+++

### いろいろなテキストエディタ
<img src="https://code.visualstudio.com/assets/updates/1_35/logo-stable.png" alt="vscode logo" title="vscode logo" width="100">
<img src="https://www.kaoriya.net/blog/2013/12/06/vimlogo-564x564.png" alt="vim logo" title="vim logo" width="100">
<img src="https://raw.githubusercontent.com/cg433n/emacs-mac-icon/master/emacs.iconset/icon_128x128.png" alt="emacs logo" title="emacs logo" width="100">
- vscode
  - いまの流行り
  - こだわりがなければこれを入れておけば大丈夫

+++

### PyTorchのインストール
#### 準備
- soxのインストール
  - Ubuntu
    ```
    sudo apt install sox libsox-dev libsox-fmt-all
    ```
  - OSX
    ```
    brew install sox
    ```

+++

### PyTorchのインストール
1. [Get Started](https://pytorch.org/get-started/locally/)から自分の環境を選択
1. Run this Command 欄に表示されたコマンドを実行
    - torch (PyTorch本体), torchvision (画像処理用パッケージ) がインストールされる
    - 仮想環境を作っている人は仮想環境を有効にしてから
1. torchaudio (音響処理用パッケージ) のインストール
    ```
    $ pip install torchaudio
    ```
1. その他パッケージのインストール
    ```
    $ pip install matplotlib torchsummary
    ```

+++

### 学習の実行
    ```
    $ cd ./src
    $ python train.py
    ```

+++

### モデルの評価
    ```
    $ python test.py
    ```

+++

### 解説
