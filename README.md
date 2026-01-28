# L_URL-Detection

一个**轻量级恶意 URL 检测**项目：从原始 URL 列表构造数据集，提取 18 个手工特征，训练传统机器学习模型（Logistic Regression 与 XGBoost），并生成评估图表、SHAP 可解释性分析与简单推理性能基准。

## 1. 项目整体流程（End-to-End）

项目的推荐主入口是：

```bash
python run_pipeline.py
```

默认会按顺序执行：

1. `data/data_preprocess.py`：从原始数据生成白名单/黑名单 URL 列表
2. `feature_extraction/run_feature_extraction.py`：特征提取 + 70/30 分层划分训练/测试集
3. `model_training/train_models.py`：训练与评估（LR 与 XGBoost），保存模型与指标
4. `model_training/generate_charts.py`：生成 ROC 曲线与混淆矩阵图
5. `SHAP/explain_models.py`：SHAP 全局/局部解释图，并做指标一致性校验
6. （可选）`efficiency_benchmark/simple_benchmark.py`：对 XGBoost 推理速度做基准测试

## 2. 目录结构与文件功能

项目已按功能拆分到子目录中；各脚本会基于自身 `__file__` 自动定位项目根目录，再拼接 `data/`、`models/`、`figures/` 等相对路径，从而避免依赖当前工作目录，保证可迁移性。

```text
L_URL-Detection/
  run_pipeline.py

  data/
    data_preprocess.py
    top-1m.csv
    verified_online.csv
    white_list.csv
    block_list.csv
    train_data.csv
    test_data.csv

  feature_extraction/
    feature_extractor.py
    run_feature_extraction.py

  model_training/
    train_models.py
    generate_charts.py
    generate_workflow_figure.py

  SHAP/
    explain_models.py
    inspect_shap_sample.py

  efficiency_benchmark/
    simple_benchmark.py
    plot_benchmark_from_json.py

  models/
    Logistic_Regression.joblib
    XGBoost.joblib

  figures/
    *.png

  evaluation_results.json
  benchmark_results.json
```

### 2.1 脚本（Python）

- **`run_pipeline.py`**
  - **功能**：一键串联项目全流程。
  - **实现方式**：用 `subprocess.run([sys.executable, script_path], cwd=project_dir, check=True)` 依次运行各步骤。
  - **参数**：
    - `--project-dir`：项目根目录（默认是 `run_pipeline.py` 所在目录）
    - `--skip-preprocess`：跳过 `data/data_preprocess.py`
    - `--skip-feature`：跳过 `feature_extraction/run_feature_extraction.py`
    - `--skip-train`：跳过 `model_training/train_models.py`
    - `--skip-charts`：跳过 `model_training/generate_charts.py`
    - `--skip-shap`：跳过 `SHAP/explain_models.py`
    - `--run-benchmark`：额外运行 `efficiency_benchmark/simple_benchmark.py`

- **`data/data_preprocess.py`**
  - **功能**：生成两个带标签的 URL 列表文件：
    - `data/white_list.csv`（良性，标签 `1`）：从 `data/top-1m.csv`（rank, domain）构造 `https://{domain}`
    - `data/block_list.csv`（恶意，标签 `0`）：从 `data/verified_online.csv` 的 `url` 列取前 `TARGET_COUNT` 条
  - **关键常量**：`TARGET_COUNT = 40000`

- **`feature_extraction/feature_extractor.py`**
  - **功能**：定义 `FeatureExtractor`，对单条 URL 提取 **18 个轻量特征**（纯字符串/解析层面，不依赖网络请求）。
  - **特征分组**：
    - 结构/长度特征（5）：
      - `url_length`, `hostname_length`, `path_length`, `dir_depth`, `filename_length`
    - 特殊符号统计（5）：
      - `count_dots`, `count_hyphens`, `has_at_symbol`, `double_slash_position`, `count_query_params`
    - 异常/混淆（4）：
      - `is_ip_address`, `is_shortened`, `https_in_hostname`, `digit_letter_ratio`
    - 语义/熵（4）：
      - `hostname_entropy`, `sensitive_word_count`, `tld_risk`, `longest_token_length`

- **`feature_extraction/run_feature_extraction.py`**
  - **功能**：
    - 读取 `data/white_list.csv` 与 `data/block_list.csv`
    - 对每个 URL 调用 `FeatureExtractor.extract_all_features(url)`
    - 合并为完整特征表后，按标签分层 `train_test_split(test_size=0.3, random_state=42, stratify=label)`
    - 输出 `data/train_data.csv` 与 `data/test_data.csv`
  - **标签约定**：`0 = Malicious`，`1 = Benign`

- **`model_training/train_models.py`**
  - **功能**：训练并评估两个模型：
    - `LogisticRegression(max_iter=1000, random_state=42)`
    - `XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)`
  - **数据**：读取 `data/train_data.csv` / `data/test_data.csv`，并 `drop(columns=['url','label'])` 作为特征。
  - **输出**：
    - 模型文件：`models/Logistic_Regression.joblib`、`models/XGBoost.joblib`
    - 指标文件：`evaluation_results.json`（list 格式，每个元素是一个模型的 metrics）

- **`model_training/generate_charts.py`**
  - **功能**：
    - 读取 `data/test_data.csv`
    - 加载训练好的模型（默认读取 `models/Logistic_Regression.joblib` 与 `models/XGBoost.joblib`）
    - 生成：
      - `figures/roc_curve.png`
      - `figures/confusion_matrix_Logistic_Regression.png`
      - `figures/confusion_matrix_XGBoost.png`

- **`SHAP/explain_models.py`**
  - **功能**：对 XGBoost 模型做可解释性分析，并做评估结果一致性校验。
  - **主要步骤**：
    - 从 `data/test_data.csv` 中构造一个 **1000 条样本子集**（默认尽量做到恶意/良性各 500，`random_state=42`）
    - `verify_evaluation_vs_current_confusion_matrix()`：用“当前 test_data + 当前模型”重新算指标，并与 `evaluation_results.json` 中的 XGBoost 行进行对比输出
    - 用 `shap.TreeExplainer(model)` 计算 SHAP：
      - 全局解释：
        - `figures/shap_summary_beeswarm.png`
        - `figures/shap_importance_bar.png`
        - `figures/shap_beeswarm_importance_combined.png`（把 beeswarm 与 bar 合并在一张图）
      - 局部解释（waterfall）：
        - `figures/shap_waterfall_sample.png`（选择一个更“典型/高置信”的恶意样本）
        - `figures/shap_waterfall_white_sample.png`（选择一个更“典型/高置信”的良性样本）
    - `crop_benchmark_left_panel()`：如存在 benchmark 结果，会额外生成/更新 `figures/benchmark_performance_specs_left.png`

- **`efficiency_benchmark/simple_benchmark.py`**
  - **功能**：对 `models/XGBoost.joblib` 的 `predict()` 做简单推理性能测试。
  - **方式**：
    - 从 `data/test_data.csv` 抽 1000 条样本
    - `warmup_rounds` 预热后重复 `repeats` 次计时
    - 用 trimmed mean 去掉极端值，报告：吞吐 TPS、平均延迟（ms/样本）等
  - **输出**：
    - `benchmark_results.json`
    - `figures/benchmark_performance_specs.png`

- **`efficiency_benchmark/plot_benchmark_from_json.py`**
  - **功能**：从 `benchmark_results.json` 重新绘图（不必重新跑 benchmark），并可尝试从 `1.txt` 读取文本摘要做一致性检查。

- **`model_training/generate_workflow_figure.py`**
  - **功能**：生成项目工作流示意图：`figures/workflow_overview.png`

- **`SHAP/inspect_shap_sample.py`**
  - **功能**：用于调试/检查 SHAP 局部样本选择：
    - 固定抽样 `data/test_data.csv` 的 1000 条
    - 找到第一条恶意样本，打印该样本的 URL、预测结果及 Top-5 贡献特征

### 2.2 数据与中间产物（CSV/JSON）

- **输入原始数据**
  - `data/top-1m.csv`：top 域名列表（脚本按 `rank,domain` 读取）
  - `data/verified_online.csv`：恶意 URL 数据（脚本按 `url` 列读取）

- **预处理输出**
  - `data/white_list.csv`：良性 URL 列表（含 `url,label`）
  - `data/block_list.csv`：恶意 URL 列表（含 `url,label`）

- **特征数据集**
  - `data/train_data.csv`：训练集（包含 `url`、`label` 以及 18 个特征列）
  - `data/test_data.csv`：测试集（同上）

- **评估与基准结果**
  - `evaluation_results.json`：训练脚本输出的评估结果（LR + XGBoost）
  - `benchmark_results.json`：性能基准脚本输出的结果（默认只 benchmark XGBoost）

## 3. 环境与依赖

建议 Python 版本：`>= 3.9`（项目中存在 `tuple[float, float]` 等语法）。

推荐使用虚拟环境，并通过 `requirements.txt` 一键安装：

```bash
pip install -r requirements.txt
```

备注：

- `shap` 依赖链较多（如 `numba` 等），首次安装可能会比较慢。
- Windows 下安装 `xgboost` 如遇到安装/运行库问题，建议优先使用官方 wheel 或 conda。

## 4. 快速开始

### 4.1 一键跑通全流程

在项目根目录执行：

```bash
python run_pipeline.py
```

如需额外跑性能基准：

```bash
python run_pipeline.py --run-benchmark
```

### 4.2 按步骤单独运行

- 生成白/黑名单：

```bash
python data/data_preprocess.py
```

- 特征提取 + 划分训练/测试：

```bash
python feature_extraction/run_feature_extraction.py
```

- 训练与评估：

```bash
python model_training/train_models.py
```

- 生成 ROC 与混淆矩阵：

```bash
python model_training/generate_charts.py
```

- 生成 SHAP 解释图：

```bash
python SHAP/explain_models.py
```

- 基准测试：

```bash
python efficiency_benchmark/simple_benchmark.py
```

- 从 JSON 重绘 benchmark 图：

```bash
python efficiency_benchmark/plot_benchmark_from_json.py
```

## 5. 输出物一览（常用）

- **模型**：`models/Logistic_Regression.joblib`、`models/XGBoost.joblib`
- **评估指标**：`evaluation_results.json`
- **图表**：`figures/roc_curve.png`、`figures/confusion_matrix_*.png`
- **解释性**：`figures/shap_*.png`
- **性能基准**：`benchmark_results.json`、`figures/benchmark_performance_specs.png`

## 6. 服务器/云端推理性能基准测试（`simple_benchmark.py`）

本项目提供了一个面向“服务器/云端部署场景”的简单推理基准测试脚本 `simple_benchmark.py`，用于评估 **XGBoost 模型在 CPU 上的批量推理吞吐与延迟稳定性**。

### 6.1 测试环境（脚本内硬编码展示）

脚本会在图中展示如下服务器配置（`SERVER_SPECS`）：

- **System**：Ubuntu 24.04
- **CPU**：2 vCPU
- **Memory**：2 GiB
- **Disk**：40 GiB
- **Bandwidth**：200 Mbps (Peak)

### 6.2 测试方法与默认参数

`simple_benchmark.py` 的默认测试方式如下：

- **模型**：读取 `models/XGBoost.joblib`，调用 `model.predict()`
- **数据**：读取 `test_data.csv`，并去掉 `url`、`label` 列后作为特征矩阵；固定抽样 `n=1000`（`random_state=42`）作为基准 batch
- **预热（Warm-up）**：默认 `warmup_rounds=50`，用于降低首次运行的冷启动影响
- **重复计时（Repeats）**：默认 `repeats=500`，每次对同一 batch 预测并用 `time.perf_counter()` 计时，得到 500 个“1000 条样本批处理”的耗时
- **去极值（Trimmed Mean）**：默认 `trim_count=10`，会在统计时丢弃最快 10 次与最慢 10 次，从而得到更稳健的平均值（减少系统抖动/偶发抢占造成的极端值影响）

### 6.3 指标解释（JSON 字段含义）

基准结果会写入 `benchmark_results.json`（列表，默认只包含 XGBoost 一项），主要字段含义：

- **`Total_Time_1000_Samples_*`**：一次批处理（1000 条样本）完成推理所需时间（单位：秒）
- **`TPS_*`**：吞吐（Transactions Per Second），按 `1000 / time_sec` 计算
- **`Avg_Latency_ms_*`**：平均单样本延迟（单位：ms），按 `(time_sec * 1000) / 1000` 计算
  - 由于固定 batch=1000，这个值通常会很小；例如 `0.00236 ms` 约等于 `2.36 µs/样本`
- **`Runs_Total_Time_Sec`**：每次重复的原始批处理耗时序列（长度≈`repeats`）

### 6.4 输出文件

- **结果 JSON**：`benchmark_results.json`
- **性能图**：`figures/benchmark_performance_specs.png`
  - 左图：批处理耗时分布（用于观察噪声/离群点）
  - 右图：TPS（原始均值 vs trimmed 均值）对比，带误差条

如需从 JSON 重新绘图（避免重跑 benchmark），可执行：

```bash
python plot_benchmark_from_json.py
```

### 6.5 本仓库已有的参考结果（来自 `benchmark_results.json`）

当前仓库中已包含一次 XGBoost 基准测试结果（你的环境重跑后数值可能不同）：

- **批处理耗时（1000 条）Trimmed Mean**：`0.0023634065 s`（约 `2.36 ms / 1000 条`）
- **吞吐 TPS Trimmed Mean**：`428068.98`（约 `428k`）
- **单样本平均延迟 Trimmed Mean**：`0.0023634065 ms`（约 `2.36 µs/样本`）
- **极端离群耗时（Max）**：`0.05183216 s`（约 `51.83 ms / 1000 条`，说明偶发抖动存在，因此使用 trimmed mean 更稳健）

## 7. 重要约定与注意事项

- **标签定义**：`0 = Malicious`，`1 = Benign`
- **运行路径**：建议始终在仓库根目录运行脚本（因为路径写死为 `./`）。
- **可重复性**：数据划分与采样大量使用 `random_state=42`。

---

如果你希望我进一步把“各脚本的参数/可配置项”做成统一的命令行接口（例如把 `TARGET_COUNT`、训练/测试比例、XGBoost 超参等都改为 CLI 参数，并补齐 `requirements.txt`），告诉我你的期望即可。
