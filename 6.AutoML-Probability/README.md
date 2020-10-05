# 6. AutoML GUI deploy Probability

Azure Machine Learning の AutoMLを使って作成したモデルを、コードレスでのACIデプロイ。その際の推論結果に Probabilityの数字が入っていません。これはカスタムスクリプトで、Probabilityも出力するサンプルです。

  - データ
      - Azure Machine Learning のチュートリアルで用意されているものを使います。
銀行のマーケテイング活動の中で、定期預金を申し込んだ人を[y]列で yes | no で設定したものです。
      - チュートリアル:Azure Machine Learning の自動 ML で分類モデルを作成する
          - https://docs.microsoft.com/ja-jp/azure/machine-learning/tutorial-first-experiment-automated-ml


# フォルダー構造


```bash
├── aci.http
├── conda_env_v_1_0_0.yml
└── scoring_file_v_1_0_0.py
```

| ファイル名                               | 内容 |
| ---------------------------------------- | ----------------- |
| [aci.http](aci.http) | Visual Studio Code のREST Client で直ぐに使えるHTTP Postの文字列 |
| [conda_env_v_1_0_0.yml](conda_env_v_1_0_0.yml) | AutoMLのジョブが作成した condaの環境設定ファイル |
| [scoring_file_v_1_0_0.py](scoring_file_v_1_0_0.py) | AutoMLのジョブが作成した 推論のPythonファイルを修正したもの |


# 参考

チュートリアル:Azure Machine Learning の自動 ML で分類モデルを作成する:
https://docs.microsoft.com/ja-jp/azure/machine-learning/tutorial-first-experiment-automated-ml