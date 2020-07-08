# Text classification using AutoML with BERT featurization in Japanese

`AutoML` は Modelの学習時における feature engineering, Hyper-parameter Turning, Job management などをまとめて行ってくれる機能になります。

その中でも、テキスト・文字列 のデータがあった際に Featurization Embedding を BiLSTMあるいは 'BERT'を使って行ってくれる機能があります。

自動機械学習 (AutoML) とは:

https://docs.microsoft.com/ja-jp/azure/machine-learning/concept-automated-ml

こちらのサンプルを少し変えただけになります。

https://github.com/Azure/MachineLearningNotebooks/tree/master/how-to-use-azureml/automated-machine-learning/classification-text-dnn

## フォルダー構造

```bash
├── data
|    └─── livedoornews
|            └─── livedoor-corpus.csv
├── auto-ml-classification-text-dnn.ipynb
├── auto-ml-classification-text-dnn.yml
├── helper.py
└── infer.py
```

| ファイル名                               | 内容 |
| ----  | ---- |
| [livedoor-corpus.csv](data/livedoornews/livedoor-corpus.csv)  | 本livedoor News のサンプルデータ              | 
| [auto-ml-classification-text-dnn.ipynb](auto-ml-classification-text-dnn.ipynb)  | AutoML の実行部分              | 
| [helper.py](helper.py) | ツール的なスクリプト |   
| [infer.py](infer.py) | 推論用 スクリプト   |  

## 参考

自動機械学習による特徴量化:

https://docs.microsoft.com/ja-jp/azure/machine-learning/how-to-configure-auto-features

How BERT is integrated into Azure automated machine learning:

https://techcommunity.microsoft.com/t5/azure-ai/how-bert-is-integrated-into-azure-automated-machine-learning/ba-p/1194657
