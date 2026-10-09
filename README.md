# ai-learning-journey

机器学习与深度学习记录。


## 进度

标注：★★★ 重点内容 / ★★☆ 懂原理会使用 / ★☆☆ 了解

### 机器学习算法 · `01_machine_learning/`

- [x] 线性回归 ★★★
- [x] 逻辑回归 ★★★
- [x] 决策树 ★★★
- [x] 集成学习（Bagging / Boosting）★★★
- [ ] XGBoost / LightGBM ★★★
- [ ] SVM ★★☆
- [ ] 朴素贝叶斯 ★★☆
- [ ] KMeans ★★☆
- [ ] PCA / LDA ★★☆
- [ ] HMM ★☆☆

### 特征工程 · `02_feature_engineering/`

- [x] 编码 / 缩放 / 文本提取 ★★★
- [x] 特征降维 ★★☆
- [ ] 财务比率特征 ★★★
- [ ] 盈余质量特征 ★★★
- [ ] 时序特征（lag / rolling / 同比）★★★
- [ ] 特征选择（IV / RFE / 树重要性）★★☆
- [ ] 缺失值 / 异常值处理 ★★☆

### 模型评估与可解释性 · `03_model_evaluation/`

- [ ] 滚动窗口验证 ★★★
- [ ] AUC / KS / F1 / Precision@K ★★★
- [ ] PSI ★★★
- [ ] SHAP ★★★
- [ ] LIME ★★☆
- [ ] 超参数调优（Optuna）★★☆

### 深度学习 · `04_deep_learning/`（待建）

- [ ] 神经网络与反向传播 ★★★
- [ ] PyTorch ★★★
- [ ] Transformer ★★★
- [ ] LSTM ★★☆
- [ ] DataLoader ★★☆
- [ ] CNN ★☆☆

### NLP 与大模型 · `05_nlp_llm/`（待建）

- [ ] Tokenizer / 分词 ★★☆
- [ ] Word2Vec / TF-IDF ★★☆
- [ ] BERT ★★★
- [ ] GPT 演进 / LLaMA / DeepSeek / MoE ★★★
- [ ] HuggingFace ★★★
- [ ] LoRA 微调 ★★★
- [ ] Prompt 工程 ★★☆
- [ ] 模型量化（GGUF）★★☆
- [ ] Ollama 本地部署 ★★☆

### RAG 与 Agent · `06_rag_agent/`（待建）

- [ ] RAG 全链路 ★★★
- [ ] 混合检索 / Reranker ★★★
- [ ] Agent（LangGraph）★★★
- [ ] MCP ★★★

### 金融专项 · `07_finance_domain/`（待建，可选）

- [ ] 金融量化 ★★☆
- [ ] 金融知识图谱 ★★☆
- [ ] 多模态 RAG ★★☆

## 不深入的方向

CV、语音、推荐系统、强化学习、机器人、医疗时序、GNN/PINN、关联规则、HMM 深入、低代码平台（Dify 除外）。

## 项目

- 上市公司财务风险预警（未开始）
- 财报研究 Agent（未开始）
- 金融 AI 可解释性与合规（未开始）
- 硕士论文：多模态风险识别（未开始）

## 目录

    01_machine_learning/      机器学习算法
    02_feature_engineering/   特征工程
    03_model_evaluation/      模型评估
    datasets/                 数据集，文件不入库
    archive/                  旧版本

## 说明

- 数据文件不入库，规则见 `.gitignore`
- 笔记以 ipynb 为主，重要结论整理成 md
- 目录和文件名用英文 snake_case，笔记正文用中文


## 环境

- macOS + PyCharm + Jupyter
- Python 3.13.5（conda base）
- 主要库：numpy, pandas, scikit-learn, matplotlib, seaborn, xgboost
