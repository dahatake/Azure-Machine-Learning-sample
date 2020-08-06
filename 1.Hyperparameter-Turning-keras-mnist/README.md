# HyperParamter Turning sample

Azure Machine Learning services の  Hyperparameters Turning を使ったサンプル。 `HyperDrive` は、`Automated Machine Learning` の機能と切り離して利用ができます。執筆時点 (2019/7/3) 時点だと、Automated Machine Learning は、Deep Learning には使えないため、単独のサンプルとして。

  - MNIST
  - Keras + Tensorflow
  - Neural Network はコードから作成
  - GPU 利用前提
    - NCv3 (Volta)
    - 複数 GPU コア利用
    - TensorCore を使うための `Automatic Mixed Precision` の有効化
    https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html

## フォルダー構造


```bash
├── train-hyperparameter-tune-deploy-with-keras.ipynb
├── keras_mnist.py
└── utils.py
```

| ファイル名                               | 内容 |
| ---------------------------------------- | ----------------- |
| [train-hyperparameter-tune-deploy-with-keras.ipynb](train-hyperparameter-tune-deploy-with-keras.ipynb)  | Azure Machine Learning Services SDK を利用し、kerasの学習ジョブを実行。HyperParameterを手動設定のものと、HyperDrive を利用したものがある              | 
| [keras_mnist.py](keras_mnist.py) | keras-TensorFlow の学習実行スクリプト |   
| [utils.py](utils.py) | Utility スクリプト   |  

## 参考

Azure Machine Learning でモデルのハイパーパラメーターを調整する:

https://docs.microsoft.com/ja-jp/azure/machine-learning/service/how-to-tune-hyperparameters

## To-Do
- Add Model Explanation