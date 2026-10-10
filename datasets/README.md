# datasets

数据文件不入库（见 `.gitignore`），这里记录**原始来源**，方便重建。

## 经典数据集

| 文件 | 数据集 | 来源 |
|---|---|---|
| `iris.csv` | Iris（鸢尾花） | R.A. Fisher, 1936 · UCI ML Repository · 也可 `sklearn.datasets.load_iris()` |
| `mnist-demo.csv` | MNIST | Yann LeCun et al., 1998 · 也可 `fetch_openml('mnist_784')` |
| `fashion-mnist-demo.csv` | Fashion-MNIST | Zalando Research, 2017 · `github.com/zalandoresearch/fashion-mnist` |
| `mnist-original.mat` | MNIST（MATLAB 格式） | 原 mldata.org 分发（该站已停服），可用 `fetch_openml('mnist_784')` 替代 |
| `microchips-tests.csv` | 微芯片质检（Microchip Test） | Andrew Ng · CS229 / Coursera ML · ex2 正则化逻辑回归 |
| `server-operational-params.csv` | 服务器运行参数 | Andrew Ng · CS229 / Coursera ML · ex8 异常检测（Latency / Throughput / Anomaly） |
| `world-happiness-report-2017.csv` | World Happiness Report 2017 | UN SDSN · Kaggle: unsdsn/world-happiness |
| `dating.txt` | 约会数据（kNN 示例） | 《机器学习实战》Peter Harrington · 第 2 章 |
| `vgsales.csv` | Video Game Sales | Kaggle: gregorut/videogamesales |
| `Pokemon.csv` | Pokemon with stats | Kaggle: abcsds/pokemon（Alberto Barradas） |
| `song_views.csv` | 用户-歌曲收听次数 | Kaggle: Million Song Dataset（同 `song_data.csv` 列名） |
| `non-linear-regression-x-y.csv` | 合成数据（249 样本，x=1~250） | 课程生成，用于演示正弦特征 + 非线性回归，无公开出处 |
