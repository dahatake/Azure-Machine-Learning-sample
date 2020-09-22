# 5.C# REST API Client For AutoML GUI Deploy To ACI

Azure Machine Learning の AutoMLを使って作成したモデルを、コードレスでのACIデプロイ。そのREST APIのコードを C# から呼び出すためのサンプル。
一般的な C#での REST API 呼び出しのサンプルとほぼ同様。

  - データ
      - Azure Machine Learning のチュートリアルで用意されているものを使います。
銀行のマーケテイング活動の中で、定期預金を申し込んだ人を[y]列で yes | no で設定したものです。
      - チュートリアル:Azure Machine Learning の自動 ML で分類モデルを作成する
          - https://docs.microsoft.com/ja-jp/azure/machine-learning/tutorial-first-experiment-automated-ml


# フォルダー構造


```bash
├── WebAPIClient
|     ├── Program.cs
|     └── WebAPIClient.csproj
└── aci.http
```

| ファイル名                               | 内容 |
| ---------------------------------------- | ----------------- |
| [Program.cs](Program.cs)  | Visual Studio Code の .NET Core で作成したコンソールアプリケーションタイプの REST Client のサンプル | 
| [aci.http](aci.http) | Visual Studio Code のREST Client で直ぐに使えるHTTP Postの文字列 |   

# 参考


Azure Functions に機械学習モデルをデプロイする (プレビュー):

https://docs.microsoft.com/ja-jp/azure/machine-learning/how-to-deploy-functions