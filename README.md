# Azure Machine Learning services Samples

`docs.microsoft.com` には、かなりの数のサンプルがあります。
幾つか、そこに無いものをこちらに置いておきます。

Azure Macine Learning services サンプルコード:
https://docs.microsoft.com/ja-jp/azure/machine-learning/service/samples-notebooks

# 事前準備

1. Azure の Subscription を作成

Azure Machine Learning services を利用するために必要です。

無料トライアル: https://azure.microsoft.com/ja-jp/free/

2. Azure Machine Learning workspace の作成

Jupyter Notebook で、[0.config.ipynb](0.config.ipynb) を実行します。

# 背景

Azure の Subscriptionを作成後、執筆時点 (2019/10/07)では、このコードから Workspace を作成してください。Azure Portal から Workspace を作成すると、CPU / GPU のデフォルトの `AmlCompute` が設定されないため、幾つかのサンプルコードが動作しないためです。勿論、作成した AmlCompute 名を直接クエリすれば、動作します。

参考: Azure Machine Learning service ワークスペースを作成する:

https://docs.microsoft.com/ja-jp/azure/machine-learning/service/setup-create-workspace

3. Jupyter Notebook から、Azure Machine Learning 参照のための設定

Azure Portal から `構成ファイル` である `config.json` をダウンロードして、この Notebook のルート直下にUploadします。

参考: 構成ファイルをダウンロードする

https://docs.microsoft.com/ja-jp/azure/machine-learning/service/how-to-manage-workspace#download-a-configuration-file

# 1. HyperParameter Turning by HyperDrive

 - [train-hyperparameter-tune-deploy-with-keras.ipynb](1.Hyperparameter-Turning-keras-mnist/README.md)

Azure Machine Learning services の  Hyperparameters Turning を使ったサンプル。 `HyperDrive` は、`Automated Machine Learning` の機能と切り離して利用ができます。執筆時点 (2019/7/3) 時点だと、Automated Machine Learning は、Deep Learning には使えないため、単独のサンプルとして。

# 2. Keras to ONNX for WindowsML

 - [train-hyperparameter-with-keras-for-WindowsML.ipynb](2.onnx-WindowsML/README.md)

`WindowsML` は、Windows 10 / Windows 2019 での推論実行に特化した WindowsのAPIです。`ONNX` のみをサポートしています。
ここでは、自身で学習・作成した `keras` のモデルを、WindowsML で実行できる形式に変換します。


# 3. Text classification using AutoML with BERT featurization in Japanese

 - [auto-ml-classification-text-dnn.ipynb](3.classification-text-dnn-jpn/README.md)

`AutoML` は Modelの学習における feature engineering, Hyper-parameter Turning, Job management などをまとめて行ってくれる機能になります。

その中でも、テキスト・文字列 のデータがあった際に Featurization Embedding を BiLSTMあるいは 'BERT'を使って行ってくれる機能があります。


# 4. Deploy AutoML model to Azure Functions (Preview)

 - [AML-AzureFunctionsPackager.ipynb](4.AML-Functions-notebook/README.md)

`Azure Functions` に Azure Machine Learning で管理されているモデルを Docker Container 化をしてデプロイするサンプルです。

# 5. REST API Client for AutoML Model deployment via Portal to ACI

 - [Program.cs](5.C#-REST-API-Client-For-AutoML-GUI-Deploy-To-ACI/README.md)

`Auto ML` を使って作成し、ACIに展開したモデル。それを、C#から呼び出すサンプルです。


## 参考

Azure Machine Learning Services ドキュメント:

https://docs.microsoft.com/ja-jp/azure/machine-learning/service/
