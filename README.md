# mHCMTN_CODE

代码仓库：[Mubaoy/mHCMTN_CODE](https://github.com/Mubaoy/mHCMTN_CODE) · 当前版本：[v0.1.1](https://github.com/Mubaoy/mHCMTN_CODE/tree/v0.1.1)

可运行的 **mHCMTN 新实现**，包含数据准备、STL-LSTM 与预测基准、需求响应优化、优化基准、测试和绘图。这里的结果需要重新运行实验获得；本项目不声称恢复了旧图对应的模型权重或历史指标。

当前优化任务：12 个等长时段，12 个离散类别 + 3 个连续响应参数。实现范围是光伏引导的负荷转移，**尚不包含电池 SOC、充放电互斥、车辆到离站时间或逐设备约束**。方法细节及文稿歧义的处理见 [METHOD.md](docs/METHOD.md)。

## 安装与快速检查

Python >= 3.10；已在 Python 3.13.9 / PyTorch 2.9.1 CPU 环境本地验证。CPU 即可运行。下面命令均在本目录执行。

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
python -m mhcmtn smoke --output results/smoke --no-plot
```

`smoke` 使用明确标注的合成数据，检查 mHCMTN、GA、PSO、MLPO、IWO、WPA 是否运行和保存结果；不能将它的指标当作论文实验。若当前运行环境不允许写系统临时目录，测试可加 `--basetemp results/pytest-temp`，请使用专门的临时目录。

## 使用真实数据

`selected_data` 的已核实来源是 `riddhi-vyas/Smart-Solar-Energy-Scheduling-with-Machine-Learning` 的原始工作簿六列子集，**不是** `solarhome-control-bench` 的两个日透视 CSV。两份数据不能互换。见 [DATA_PROVENANCE.md](docs/DATA_PROVENANCE.md)。仓库不附原始 Excel；可以下载固定版本，或指定自己的本地原文件。

```powershell
# 二选一：固定版本下载，或使用已有 selected_data.xlsx / 原始工作簿
python -m mhcmtn prepare-data --download --output results/prepared
python -m mhcmtn prepare-data --source "D:/your_data/selected_data.xlsx" --output results/prepared

# 默认训练 STL-LSTM、LSTM、Random Forest、XGBoost、SVR
python -m mhcmtn forecast --data-dir results/prepared --output results/forecast --config configs/default.json

# 使用测试期案例中的 12 步递归 STL-LSTM 预测进行优化
python -m mhcmtn optimize --case results/forecast/planning_case_stl_lstm.json --output results/mhcmtn --config configs/default.json

# 相同预测、解码器和目标函数下的六种优化方法
python -m mhcmtn benchmark --case results/forecast/planning_case_stl_lstm.json --output results/benchmark --config configs/default.json

# 用保存的最优可行决策重算结果
python -m mhcmtn check --checkpoint results/mhcmtn/checkpoint.pt
```

也可以一条命令串联下载、准备、预测和优化：

```powershell
python -m mhcmtn pipeline --download --work-dir results/full --config configs/default.json
```

`pipeline` 默认仅优化 mHCMTN；增加 `--methods mhcmtn ga pso mlpo iwo wpa` 可运行全部优化方法。一次快速的真实数据检查可以加 `--forecast-epochs 1 --epochs 20 --iterations 10 --no-plot`。这会减少训练量，应与完整实验分开记录。`--epochs` 控制神经优化步数；`--forecast-epochs` 控制预测模型训练轮数，两者含义不同。

## 模型与参数

| 模块 | 默认设置 |
|---|---|
| STL-LSTM | 因果端点 STL，周期 24 / 窗口 168；3 个分量分支，各两层 50 单元 LSTM；24 小时回看 |
| 预测训练 | Adam 0.001，batch 32，最多 50 epochs，验证集早停 patience 10 |
| mHCMTN | width 32 / heads 4 / layers 2 / FFN 64；45 个潜在 token；dropout 0.1 |
| mHCMTN 优化 | Adam 0.001，1000 步，单实例 batch 1；温度 1.0 → 0.1 |
| GA / PSO / IWO / WPA | 初始种群 30，1000 次外层迭代；详细参数在 JSON 中 |
| MLPO | 15→15→15，Tanh，Adam 0.01，1000 步 |

[configs/default.json](configs/default.json) 包含全部优化参数和神经预测参数；[configs/manuscript_width.json](configs/manuscript_width.json) 使用 64 / 8 / 2 / 256 宽度，其余字段采用默认值。RF/XGBoost/SVR 的设置在 [forecasting.py](mhcmtn/forecasting.py) 中显式声明并写入运行记录。

这些是新实现中实际生效的配置。此前拟定的“虚拟超参数”不能作为已运行实验的证据；论文应以实际运行保存的 `configuration.json`、`result.json` 和环境信息为准。

## 输出与复现约定

- 数据准备：`prepared.npz`、`split.json`、`metadata.json`。时间序列按 UTC 保持连续，原始当地时间另外保存；不丢弃夏令时重复时段。
- 预测：训练历史、权重、scalers、真实测试值与预测 CSV、单步测试误差 `metrics.json`、12 步递归误差 `planning_metrics.json`，以及各模型的规划输入。
- 优化：`result.json`、`history.csv`、`schedule.csv`、`environment.json`；神经方法另存 checkpoint。
- 绘图：直接读取实际运行结果，Times New Roman、18 pt，PNG 600 dpi，另存 SVG/PDF；字体需在运行机器上安装。图片不是对旧曲线的拟合。
- `checkpoint.pt` 分别保存最后模型权重和搜索到的最优决策。重现报告的调度结果使用其中的 `best_decision`，无需重新采样 Gumbel 噪声。

所有预测误差均对原始观测计算。优化使用预测光伏，真实未来光伏只用于事后评估。单步测试与 12 步递归规划是两种不同协议，详见 [METHOD.md](docs/METHOD.md)。相同迭代次数不等于相同评估预算，比较 CSV 保存各方法实际目标函数评估次数与耗时。

## 目录与发布

```text
mhcmtn/       数据、预测、调度解码器、mHCMTN、优化算法、绘图与命令行
configs/      默认参数及文稿宽度参数
data/splits/  5856 行数据的新时间顺序划分（目标行号，从 0 开始）
examples/     离线合成案例
tests/        能量守恒、梯度、时序泄漏、保存/加载等测试
docs/         方法、来源、实现计划、验证记录和可用性声明模板
.github/      上传后运行的 CPU 测试工作流
```

运行产生的数据、权重和结果由 `.gitignore` 排除。许可证状态见 [LICENSE_NOTICE.md](LICENSE_NOTICE.md)，仓库维护说明见 [PUBLISHING.md](docs/PUBLISHING.md)。已填入仓库链接的代码可用性声明见 [CODE_AVAILABILITY.txt](CODE_AVAILABILITY.txt) 和 [AVAILABILITY_TEMPLATE.md](docs/AVAILABILITY_TEMPLATE.md)。

已完成 24 项本地测试、真实数据上的五模型短程预测、六种优化器的默认 1000 步检查，以及 checkpoint 恢复。完整记录与实验边界见 [VALIDATION.md](docs/VALIDATION.md)，独立代码检查记录见 [CODE_REVIEW.md](docs/CODE_REVIEW.md)，版本变化见 [CHANGELOG.md](CHANGELOG.md)。

English summary: This repository contains a newly implemented, instance-specific
manifold-aware hyperconnection optimizer and causal STL-LSTM forecasting pipeline.
It supplies feasible energy-conserving demand-response decoding, five forecasting
methods, six optimizers, reproducibility metadata and tests. It does not reproduce
historical manuscript results, implement battery/EV scheduling, or establish
comparative performance through the included software checks.
