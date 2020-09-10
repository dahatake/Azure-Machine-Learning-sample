# Deploy AutoML model to Azure Functions (Preview)

Azure Machine Learning の ML Studio の UI 上の Auto ML で作成したモデルを、Azure Functions に展開するためのサンプル。

  - データ
      - Azure Machine Learning のチュートリアルで用意されているものを使います。
銀行のマーケテイング活動の中で、定期預金を申し込んだ人を[y]列で yes | no で設定したものです。
      - チュートリアル:Azure Machine Learning の自動 ML で分類モデルを作成する
          - https://docs.microsoft.com/ja-jp/azure/machine-learning/tutorial-first-experiment-automated-ml


# フォルダー構造


```bash
├── AutoML1bb3ebb0477 
|     ├── conda_env_v_1_0_0.yml
|     ├── model.pkl
|     └── scoring_file_v_1_0_0.py
└── AML-AzureFunctionsPackager.ipynb
```

| ファイル名                               | 内容 |
| ---------------------------------------- | ----------------- |
| [AML-AzureFunctionsPackager.ipynb](AML-AzureFunctionsPackager.ipynb)  | Azure Machine Learning のPreview SDK を利用し、Azure Functions の HTTP Trigger 用の docker container を作成する | 
| [AutoML1bb3ebb0477](AutoML1bb3ebb0477) | ML Studio のUIのモデルの画面から ダウンロードしたzipファイルを解凍したもの |   

# 参考


Azure Functions に機械学習モデルをデプロイする (プレビュー):

https://docs.microsoft.com/ja-jp/azure/machine-learning/how-to-deploy-functions