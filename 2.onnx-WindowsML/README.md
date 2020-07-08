# Keras Train and Convert to ONNX for WindowsML

`WindowsML` は、Windows 10 / Windows 2019 での推論実行に特化した WindowsのAPIです。`ONNX` のみをサポートしています。
ここでは、自身で学習・作成した `keras` のモデルを、WindowsML で実行できる形式に変換します。

## フォルダー構造

```bash
├── train-hyperparameter-with-keras-for-WindowsML.ipynb
├── mnist_cnn.py
└── utils.py
```

| ファイル名                               | 内容 |
| ---------------------------------------- | ----------------- |
| [学習ジョブ実行](train-hyperparameter-with-keras-for-WindowsML.ipynb)  | Azure Machine Learning Services SDK を利用し、kerasの学習ジョブを実行。HyperParameterを手動設定のものと、HyperDrive を利用したものがある              | 
| [mnist_cnn.py](mnist_cnn.py) | keras-TensorFlow の学習実行スクリプト |   
| [utils.py](utils.py) | Utility スクリプト   |  

## 参考

Windows ML ドキュメント:

https://docs.microsoft.com/ja-jp/windows/ai/windows-ml/

Windows ML ツールとサンプル:

https://docs.microsoft.com/ja-jp/windows/ai/windows-ml/tools-and-samples

Azure Machine Learning services - ONNX サンプル:

https://aka.ms/onnxnotebooks

## To-Do
- Add Model Explanation