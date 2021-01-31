# Azure Machine Learning - Image Classification HyperParamter Tuning - Pytorch Lightning

Azure Machine Learning の HyperDrive を使った Hyperparameters Turning のサンプル。 `HyperDrive` は、`Automated Machine Learning` の機能と切り離して利用ができます。
ここでは Image Classificaiton を扱います。

## 実装している主な機能
- Fine Turning
    - Transfer Learning も選べます
- Data augmentation
- Hyperparameter Tuning

## To-Do😅
- Confusion Matrix

# 環境

  - Python 3.6
  - Anaconda
  - Dataset: 任意のものを
      - 例えば...カルビーさんのポテトチップスの写真
          - https://github.com/DeepLearningLab/PotatoChips-Classification
  - 学習環境は GPU の方がいいです。

Jupyter Notebook の実行はローカルのコンピューターでも可能です。その際は、開発環境として以下のパッケージを入れてください。

```bash
name: azureml-pl-dev
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.6
  - matplotlib
  - jedi==0.17.1
  - jupyterlab
  - pip
  - pip:
    - tornado==6.1.0
    - azureml
    - azureml-core
    - azureml-sdk
    - azureml-widgets
    - torch
    - torchvision
    - pytorch-lightning
```

anaconda の Terminal から
```bash
conda install notebook ipykernel
ipython kernel install --user --name <myenv> --display-name "Python (myenv)"
```

参考:

Azure Machine Learning のために Python 開発環境をセットアップする:

https://docs.microsoft.com/ja-jp/azure/machine-learning/how-to-configure-environment#jupyter-notebooks

# フォルダー構造

```bash
├── .amlignore
├── environment.yml
├── environment_dev.yml
├── ImageClassification-hyperparameterTune-PyTorchLightning.ipynb
├── keras_mnist.py
└── train.py
```

| ファイル名                               | 内容 |
| ---------------------------------------- | ----------------- |
| .amlignore | Azure Machine Learning が各ジョブの実行時に学習ノードにコピーしないファイルの指定 |
| environment.yml | Anaconda の環境作成ファイル。学習環境用 |
| environment_dev.yml | Anaconda の環境作成ファイル。開発環境用。 |
| [ImageClassification-hyperparameterTune-PyTorchLightning.ipynb](ImageClassification-hyperparameterTune-PyTorchLightning.ipynb)  | Azure Machine Learning Services SDK を利用し、PyTorch の学習ジョブを実行。手動設定のものと、HyperDrive を利用したものがある              | 
| [train.py](train.py) | keras-TensorFlow の学習実行スクリプト |   

# 開発環境構築

1. Anaconda のインストール
2. conda 環境の作成

```bash
conda env create -f environment.yml
```
3. Azure Blob Storage に任意のファイルをアップロード。Azure Storage Explorer での作業がおススメ😊
    https://azure.microsoft.com/ja-jp/features/storage-explorer/

Blob Storage 上の ファイル構造

TorchVision では、フォルダー名をそのままクラス名として利用できます。

猫と犬の例:

```bash
cat_dog
├── cat
|    ├── 0.png
|    ├── 1.png
|    ├── 2.png
|    └── ...
└── dog
    ├── 0.png
    ├── 1.png
    ├── 2.png
    └── ...
```
4. Azure Machine Learning の Studio で、GPUの使える Aml Compute Cluster を作成。
    作成後、以下のセルの **`name`** を置き換えます。 

```python
compute_target = ComputeTarget(workspace=ws, name='gpucluster6')
compute_target
```

    参考:

    Azure Machine Learning コンピューティング クラスターの作成

    https://docs.microsoft.com/ja-jp/azure/machine-learning/how-to-create-attach-compute-cluster?tabs=python

5. Azure Machine Learning の Studio で、Data Store / Dataset の作成

    作成後、以下のセルの **`name`** を置き換えます。 

```python
dataset = Dataset.get_by_name(ws, name='cat_dogs')
```

    参考:
    Azure Machine Learning スタジオを使用してデータに接続する

     https://docs.microsoft.com/ja-jp/azure/machine-learning/how-to-connect-data-ui


6. Azure Portal (portal.azure.com) から Azure Machine Learning の Workspace 認証用の config.json をダウンロード。この作業フォルダーにコピー

# 実行

[ImageClassification-hyperparameterTune-PyTorchLightning.ipynb](ImageClassification-hyperparameterTune-PyTorchLightning.ipynb) の各セルを実行してください。

- Hyperparameter の Sampling は、まずは Random Sampling で比較的大き目の範囲と長めの学習時間で、あたりをつけます。

例:
```python
ps = RandomParameterSampling(
    {
        '--batch-size': choice(50, 100),
        '--epoch': choice(1, 2),
        '--learning-rate': loguniform(-4, -1),
        '--momentum': loguniform(-3, -1),
        '--model-name': choice('resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception'),
        '--optimizer': choice('SGD','Adagrad','Adadelta','Adam','AdamW','Adamax', 'ASGD', 'RMSprop', 'Rprop'),
        '--criterion': choice('cross_entropy')
    }
)
```

その後、Bayesian Sampling で絞っていきます。

例:
```python
ps = BayesianParameterSampling(
    {
        '--batch-size': choice(50, 100, 150, 200, 250, 300),
        '--epoch': choice(20, 25, 30, 35),
        '--learning-rate': loguniform(-4, -1),
        '--momentum': loguniform(-2, -1),
        '--model-name': choice('resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception'),
        '--optimizer': choice('SGD','Adagrad','Adadelta','Adam','AdamW','SparseAdam', 'Adamax', 'ASGD', 'RMSprop', 'Rprop'),
        '--criterion': choice('cross_entropy', 'binary_cross_entropy', 'binary_cross_entropy_with_logits', 'poisson_nll_loss', 'hinge_embedding_loss', 'kl_div', 'l1_loss', 'mse_loss', 'margin_ranking_loss', 'multilabel_margin_loss', 'multilabel_soft_margin_loss', 'multi_margin_loss','nll_loss', 'smooth_l1_loss', 'soft_margin_loss')
    }
)

```

前の HyperDrive の実行状態を加味したジョブ実行もできます。


ハイパーパラメーター調整をウォーム スタートする (省略可能):
https://docs.microsoft.com/ja-jp/azure/machine-learning/how-to-tune-hyperparameters#warm-start-hyperparameter-tuning-optional

7. 学習結果を Azure Machine Learning Studio (ml.azure.com) で確認する。

![hyperdrive](.\img\hyperdrive.jpg)

- `並列処理グラフ` の使い方
    - 縦線の範囲をドラッグアンドドロップすると、フィルタリングが出来ます
    - 列名もドラッグアンドドロップして、左右の入れ替えが出来ます
    - 本日時点で、拡大画面ですと上記は動きません。バグとして認識されていますので、近い将来更新される予定です。

# 参考

Azure Machine Learning でモデルのハイパーパラメーターを調整する:

https://docs.microsoft.com/ja-jp/azure/machine-learning/service/how-to-tune-hyperparameters

Azure Machine Learning を使用して PyTorch モデルを大規模にトレーニングする:

https://docs.microsoft.com/ja-jp/azure/machine-learning/how-to-train-pytorch

Training Your First Distributed PyTorch Lightning Model with Azure ML:

https://medium.com/microsoftazure/training-your-first-distributed-pytorch-lightning-model-with-azure-ml-f493d370acb


FINETUNING TORCHVISION MODELS:

https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html

Split data with PyTorch Transformer:

https://stackoverflow.com/questions/61811946/train-valid-test-split-for-custom-dataset-using-pytorch-and-torchvision

