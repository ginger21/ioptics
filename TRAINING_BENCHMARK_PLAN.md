# ioptics 训练与求导方法综合基准评测方案 (Training Methods Benchmark Plan)

本文档定义了 `ioptics` 框架支持的各类集成光神经网络（IONN）训练与物理求导方法的系统性 Benchmark 方案。本方案聚焦于**真实分类任务的准确率（Classification Accuracy）**、**计算与光学测量效率（Efficiency）** 以及 **硬件非理想性抗噪鲁棒性（Hardware Robustness）**，为算法选型、框架性能优化与学术成果提供严格、可复现的标准化量化标尺。

---

## 一、评测背景与核心目标

目前仓库中的 `benchmarks/benchmark_baseline.py` 聚焦于理想无噪条件下单步算子级（转移矩阵构建、前向传播、单次反向求导）的执行耗时。为了全面评估各求导方法在真实分类任务与片上物理硬件环境下的性能，本方案扩展构建**端到端训练方法综合评测体系**，重点回答以下四个核心科学与工程问题：

1. **计算与内存扩展性**：随着波导端口数 $N$、批大小 $B$ 扩展，各求导策略的单步训练延迟与显存/内存开销如何演化？
2. **分类精度与物理光学测量效率（Accuracy vs. Optical Passes）**：在真实生物与图像分类任务上达到相同分类准确率（如 $85\%$ / $90\%$）时，各方法分别需要多少训练迭代（Epochs）、多少数字计算时间（Wall-clock Time），以及多少次**物理片上光学前向测量（Optical Forward Passes）**？
3. **空间热串扰与模型失配适应性**：当光芯片存在不可忽略的波导间热串扰（Thermal Crosstalk）时，离线盲训直接部署（In-Silico to Hardware）、串扰感知校准（Crosstalk-Aware Calibration）与免建模原位调谐（Model-Free In-Situ Tuning）在测试集准确率保持率上有何本质差距？
4. **全链路光电噪声与量化抗噪鲁棒性**：在激光器 RIN/相位噪声、光电探测器散粒/热噪声及有限比特 ADC 量化截断下，哪些求导算法表现最稳健？ADC 最小量化步长与微扰步长之间存在的“量化死区”边界如何影响原位梯度估计？

---

## 二、参评求导方法全景矩阵与物理假设

`ioptics` 统一评测以下 **5+1 种核心方法**。在核算物理测量成本时，严格区分**“仅末端强度可测（End-Point Only）”**与**“需内部原位干涉探测或纯数字计算（Intermediate In-Situ / Digital Twin）”**两类物理前提：

| 编号 | 方法名称 | 接口类型与核心类 | 单步光学前向测量次数 (设总可训参数量为 $P$) | 物理实现前提 | 算法定位与数学特性 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **M1** | **Default (Autograd)** | `BackwardType.DEFAULT` | $1$ 次（纯数字参考） | 纯数字计算机模型 | 理想数字孪生精度与速度上限基线，依赖完整转移矩阵可微计算图 |
| **M2** | **AdjointVariant (伴随场法)** | `BackwardType.ADJOINT_VARIANT`<br>(`AdjointStrategy`) | $2$ 次（正向场 + 伴随误差场反传） | 芯片内部节点原位干涉探测 / 互易注入 | 理论最优全光原位梯度法，计算复杂度 $O(1)$，但物理上要求各级相移器处具备就地干涉探测能力 |
| **M3a** | **ParameterShift (算子级)** | `BackwardType.PARAMETER_SHIFT`<br>(`ParameterShiftStrategy`) | $O(P)$（单层列内向量化 $+\pi/2$ 移相） | 混合计算图支持 | 验证底层算子批处理并行化性能与解析移相律的数值一致性 |
| **M3b** | **ParameterShift (模型级)** | `ParameterShiftDifferentiator` | $2P + 1$ 次（全网 $\pm \pi/2$ 双向移相） | **仅末端强度探测**（非侵入式） | 片上原位精确求导标准，无需芯片内部节点探测，通过宏观 $\pm\pi/2$ 移相天然抵抗微小加性噪声 |
| **M4** | **Bernoulli (理想 SPSA)** | `BernoulliDifferentiator` | **恒定 $2$ 次**（全网同步 $\pm\Delta$ 伯努利随机扰动） | **仅末端强度探测**（非侵入式） | 大规模参数规模（$P \gg 100$）下的低测量复杂度黑盒优化基线 |
| **M5** | **PhysicalBernoulli (硬件在环 SPSA)** | `PhysicalBernoulliDifferentiator` + `Simulator` | **恒定 $2$ 次**（含物理噪声与量化模拟） | **仅末端强度探测**（非侵入式） | 真实光子芯片免建模（Model-Free）闭环自整定基准 |

> **扩展对比项（M3c - Noisy ParameterShift）**：在 Level 3 硬件非理想性评测中，引入基于硬件仿真输出的参数位移求导（`NoisyParameterShiftDifferentiator`），与 `PhysicalBernoulliDifferentiator` 在含噪环境下直接对比分类精度的抗噪特性。

---

## 三、真实标准分类任务集体系 (Classification Task Suite)

全面废除纯人工合成的抽象散点数据，全部采用具有明确物理分类决策边界的真实数据集，并按端口规模 $N$ 阶梯式布局：

```text
4 端口 (小规模)          8 端口 (中规模)          16 端口 (大规模)
[Iris (3-Class)]  -->   [Digits-8 (4-Class)] --> [Digits-16 / MNIST (4-Class)]
[XOR (2-Class)]
```

| 任务标识 | 数据集名称 | 端口数 $N$ | 类别数 / 样本量 | 物理编码与特征说明 | 依赖性 | 任务核心目标 |
| :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| `iris` | **Iris (鸢尾花)** | **$N=4$** | **3 类**<br>150 样本 (120 训 / 30 测) | • 原生 4 个连续生物特征（萼片/花瓣长宽），与 **4 端口波导天然 1:1 对齐**<br>• 恒定光强归一化至 $[0, 2.0]\,\text{mW}$ | **本地原生**<br>(`sklearn.datasets.load_iris`) | 评估小规模光芯片对真实生物特征的分类准确率与收敛效率 |
| `xor` | **XOR (四象限分类)** | **$N=4$** | **2 类**<br>160 样本 (128 训 / 32 测) | • 二维输入 $(x_1, x_2)$ 补齐至 4 端口，恒定总功率注入 $P_0=10$<br>• 严格非线性分类边界 | **本地生成**<br>(数学确定性生成) | 严格检验双层非线性激活对线性不可分问题的分类决策能力 |
| `digits8` | **Digits-8 (手写数字)** | **$N=8$** | **4 类** (数字 0,1,2,3)<br>约 720 样本 (576 训 / 144 测) | • 真实手写数字图像自适应池化至 8 维连续特征<br>• 具有真实图像的空间流形与光敏特征 | **本地原生**<br>(`sklearn.datasets.load_digits`) | 评估中等规模端口下的多分类性能与参数位移测量开销 |
| `digits16` / `mnist` | **Digits-16 / MNIST-4Class** | **$N=16$** | **4 类** (数字 0,1,2,3)<br>约 800~1000 样本 | • 图像降采样至 $4\times 4=16$ 像素，每个像素对应一个波导端口输入功率<br>• IONN 领域最权威通用的标准图像分类基准 | **本地原生** (Digits-16)<br>**可选下载** (MNIST) | 评估大规模端口下的分类泛化性能、多层级联插损与热串扰影响 |

---

## 四、网络模型拓扑与非线性分类架构

针对分类任务的非线性决策需求，统一构建**双层非线性光分类网络**（可退化为单层用于线性基准对比）：

$$\boldsymbol{x} \in \mathbb{R}_+^N \xrightarrow[\text{功率注入}]{\text{电光编码}} \boldsymbol{E}_{\text{in}} \in \mathbb{C}^N \xrightarrow{\text{ClementsMesh}_1} \boldsymbol{E}_1 \xrightarrow{\text{ElectroOpticActivation}} \boldsymbol{E}_2 \xrightarrow{\text{ClementsMesh}_2} \boldsymbol{E}_{\text{out}} \xrightarrow{\text{AbsSquared}} \boldsymbol{I}_{\text{det}} \in \mathbb{R}_+^N \xrightarrow{\text{CrossEntropy}} \mathcal{L}$$

* **核心参数配置**：
  * **主干结构**：`ClementsMesh(N) -> ElectroOpticActivation(alpha=0.1) -> ClementsMesh(N) -> AbsSquared()`；
  * **损失函数**：`torch.nn.CrossEntropyLoss`，以输出端口的前 $K$ 个强度值直接作为类别非归一化对数概率（Logits）；
  * **评估指标**：Top-1 预测准确率 $\text{Accuracy} = \frac{1}{M}\sum_{i=1}^M \mathbb{I}(\arg\max(\boldsymbol{I}_i) = y_i) \times 100\%$。

---

## 五、三级递进式评测结构设计 (Three-Tier Experimental Structure)

```text
benchmarks/
├── __init__.py
├── benchmark_baseline.py             # [已有] 底层算子级无噪耗时基准
├── benchmark_training_methods.py     # [主程序] 综合训练评测 CLI (支持 --level 1/2/3/all 与 --quick)
├── tasks.py                          # [任务工厂] 标准分类数据集加载器 (Iris, XOR, Digits-8, Digits-16) 与模型构建
├── metrics.py                        # [指标与适配器] 统一求导适配器、链式法则 NoisyPS、测量计数器 (OpticalPassTracker)
├── plot_benchmark_results.py         # [可视化] 自动解析 JSON 生成 4 张论文级对比图表
└── results/                          # 结构化评测结果输出目录
    ├── level1_scalability_fidelity.json
    ├── level2_classification_convergence.json
    └── level3_hardware_robustness.json
```

### Level 1：算子扩展性与梯度保真度微基准 (`run_level1_micro_benchmark`)
1. **实验 1A（单步训练耗时与内存开销扩展性）**
   - 扫描端口规模 $N \in \{4, 8, 16, 32\}$（`quick` 模式下仅测试 $N \in \{4, 8\}$），Batch Size $B \in \{1, 32\}$；
   - 测定 M1 (`Default`)、M2 (`AdjointVariant`)、M3a (`ParameterShiftStrategy`)、M3b (`ParameterShiftDifferentiator`)、M4 (`Bernoulli`)、M5 (`PhysicalBernoulli`) 完成单步完整参数更新的中位数延迟（`median_ms`）与显存开销。
2. **实验 1B（梯度估计保真度与超参敏感度）**
   - 以高精度 `Default` 解析梯度为基准，验证 M2、M3a、M3b 的余弦相似度（验证 $> 0.9999$）；
   - 扫描 `Bernoulli` 在不同扰动步长 $\Delta \in \{0.001, 0.01, 0.05, 0.1, 0.25\}$ 与采样平均次数 $K \in \{1, 4, 16, 64\}$ 下的余弦相似度与方差。

### Level 2：真实分类任务收敛速度与光学测量复杂度 (`run_level2_convergence_benchmark`)
1. **实验 2A（标准分类任务端到端收敛轨迹）**
   - 在 `Iris` ($N=4$)、`Digits-8` ($N=8$)、`Digits-16` ($N=16$) 任务上，对比各求导方法的训练过程；
   - 所有方法共享**完全相同的初始参数权重快照（`initial_state_dict`）**、相同的数据 Shuffle 顺序及固定随机种子（默认跑 3 个独立种子）；
   - 记录每一轮 Epoch 的元组：`(epoch, wall_time_sec, cumulative_optical_passes, train_loss, test_accuracy)`。
2. **三坐标轴收敛评估体系**：
   - **Accuracy vs. Epochs**：检验单步梯度的确定性与有效优化方向；
   - **Accuracy vs. Wall-clock Time (s)**：检验数字仿真器上的综合计算效率；
   - **Accuracy vs. Optical Forward Passes**：核心物理指标，评估达到 $85\%$ / $90\%$ 分类准确率所需的真实物理片上测量总次数，量化 `Bernoulli`（$2$ 次/步）与 `ParameterShift`（$2P$ 次/步）的**效率盈亏平衡点**。

### Level 3：硬件非理想性与原位调谐鲁棒性基准 (`run_level3_robustness_benchmark`)
1. **实验 3A（空间热串扰三范式压力测试）**
   - 在 `Digits-8` ($N=8$) 任务上，扫描相邻相移器热串扰系数 $\alpha \in \{0.0, 0.02, 0.05, 0.10, 0.15\}$；
   - 对比三类范式在含热串扰硬件上的**测试集分类准确率保持率（Accuracy Retention）**：
     * **范式 A（离线盲训直接部署）**：无串扰理想模型上训练，直接部署至含热串扰硬件；
     * **范式 B（串扰感知校准训练）**：模型已知底层热扩散矩阵 $C$，通过 $C^\top \nabla_{\boldsymbol{\theta}_{\text{eff}}} \mathcal{L}$ 进行梯度补偿训练；
     * **范式 C（免建模原位闭环训练）**：完全未知 $C$ 矩阵，直接使用 `PhysicalBernoulli` 在含串扰硬件输出上闭环更新。
2. **实验 3B（ADC 量化死区与光电噪声扫描）**
   - 在 `Iris` ($N=4$) 或 `Digits-8` ($N=8$) 上配置 `HardwareProfile.LAB`；
   - 二维网格扫描 `(adc_bits, step_size)`，其中 $\text{adc\_bits} \in \{4, 6, 8, 10, 12\}$，$\text{step\_size} \in \{0.005, 0.02, 0.05, 0.15\}$；
   - 观测 `auto_gain=True` 对防止信号落入 ADC 量化死区（Dead-Zone）的增益补偿效果。
3. **实验 3C（多层级联插损容忍度测试）**
   - 在双层光网络中开启 `cascaded_loss=True`，扫描列插损 $\text{layer\_loss\_db} \in \{0.0, 0.2, 0.5, 1.0\}\,\text{dB}$ 对深层网络最终分类准确率的影响。

---

## 六、关键理论修正与工程实现防坑指南 (Crucial Implementation Rules)

在编写各模块代码时，必须严格遵守以下 5 条准则：

### 1. 【核心修正】含噪参数位移求导严格遵守复合函数链式法则
* **禁止**对整个分类损失函数直接使用 $\frac{\mathcal{L}_+ - \mathcal{L}_-}{2}$，因为 $\mathcal{L}$ 对 $\theta$ 非正弦依赖；
* **必须**采用**“硬件位移输出 + 软件链式反传”**：
  $$\frac{\partial \mathcal{L}}{\partial \theta_i} = \sum_{j=1}^N \frac{\partial \mathcal{L}}{\partial y_j} \cdot \frac{y_j(\theta_i + \pi/2) - y_j(\theta_i - \pi/2)}{2}$$
  其中 $\boldsymbol{y}_+, \boldsymbol{y}_-$ 为 `simulator.run` 的物理测量输出，$\frac{\partial \mathcal{L}}{\partial \boldsymbol{y}}$ 由数字端 PyTorch 自动求导提供。

### 2. 精确剥离“求导物理测量”与“日志评估测量”
* 统计“累计光学前向测量次数（Optical Forward Passes）”时，计数器**仅累加求导过程产生的物理前向传播**；
* 每轮 Epoch 结束时用于监控模型精度的 `evaluate(model, test_loader)` 调用必须标记为评估用途，严禁计入求导物理测量总数。

### 3. 严格遵守非负光功率与复数域约束
* 当接入 `Simulator` 时，输入张量必须严格非负（$\ge 0$），否则触发 `ValueError`；
* 探测器层末端必须包含 `AbsSquared()` 转换为实数光强，再接入 `CrossEntropyLoss`。

### 4. 保证跨方法初始权重与数据 Shuffle 严格一致
* 对比不同训练方法时，首先初始化基准模型并导出 `initial_state_dict = copy.deepcopy(model.state_dict())`；
* 每个方法开跑前调用 `model.load_state_dict(copy.deepcopy(initial_state_dict))` 并重置随机数种子 `torch.manual_seed(seed)`。

### 5. 避免适配器引入不必要的显存拷贝
* 统一适配器 `AutogradDifferentiator` 主要用于 Level 2 的多方法公平训练调度；在 Level 1 测试“原生算子级延迟”时，保留直接调用 `loss.backward()` 的纯净测试路径，防止张量拼接开销劣化基线数据。

---

## 七、逐文件详细实现规范 (File-by-File Specification)

### 文件 1：`benchmarks/metrics.py`
- **`AutogradDifferentiator(Differentiator)`**：
  封装计算图求导（`Default` 与 `AdjointVariant`），在 `differentiate` 中执行前向、`loss.backward()`，并提取展平的可训梯度张量返回。
- **`NoisyParameterShiftDifferentiator(Differentiator)`**：
  基于真实 `Simulator` 的参数位移微分器，严格按照链式法则实现 $\nabla_{\boldsymbol{\theta}} \mathcal{L} = (\nabla_{\boldsymbol{y}} \mathcal{L}) \cdot \frac{\boldsymbol{y}_+ - \boldsymbol{y}_-}{2}$。
- **`MultiSampleBernoulliDifferentiator(Differentiator)`**：
  包装 `BernoulliDifferentiator`，单步重复采样 $K$ 次独立扰动向量并对梯度求平均，抑制随机梯度方差。
- **`OpticalPassTracker`**：
  维护 `cumulative_diff_passes`（求导物理前向次数）与 `cumulative_eval_passes`（评估前向次数），按方法类型（`default: 1`, `adjoint: 2`, `param_shift: 2P+1`, `bernoulli: 2K`）精确递增。
- **`compute_gradient_metrics(grad_est, grad_ref)`**：
  计算估计梯度与参考梯度的余弦相似度（`cosine_similarity`）、相对 $L_2$ 误差（`rel_l2_error`）及梯度范数。

### 文件 2：`benchmarks/tasks.py`
- **`get_iris_task(seed=42)`**：
  加载 `load_iris`，4 维特征，3 分类，标准化至 $[0, 2.0]$ 光功率范围，划分训练集（120）与测试集（30），返回 `TaskBundle`。
- **`get_xor_task(seed=42)`**：
  生成二维 XOR 四象限数据，补齐为 4 端口光功率输入，返回 2 分类 `TaskBundle`。
- **`get_digits8_task(seed=42)`**：
  加载 `load_digits`，筛选 0, 1, 2, 3 四类，池化至 8 维特征，归一化光功率，返回 `TaskBundle`。
- **`get_digits16_task(seed=42)`**：
  加载 `load_digits`，筛选 0, 1, 2, 3 四类，重采样至 $4\times 4=16$ 维特征，归一化光功率，返回 `TaskBundle`。
- **`build_optical_classifier(num_ports, num_classes, depth=2, ...)`**：
  构建 `ClementsMesh -> ElectroOpticActivation -> ClementsMesh -> AbsSquared` 分类模型，支持配置热串扰系数与插损。
- **`build_method_suite(...)`**：
  依据统一模型模板，克隆生成 6 组完全同构、共享初始参数权重的 `(model, differentiator, method_name)` 实例。

### 文件 3：`benchmarks/benchmark_training_methods.py`
- **CLI 参数**：
  - `--level`：`1`, `2`, `3`, `all`（默认 `all`）；
  - `--quick`：快速冒烟开关（开启后缩小端口规模与 Epoch 数，15 秒内跑通整个流水线，供 CI 校验）；
  - `--output-dir`：默认 `benchmarks/results`。
- **子实验实现**：
  - `run_level1_scalability_and_fidelity(...)`：执行实验 1A 与 1B，输出延时、内存与梯度相似度数据；
  - `run_level2_convergence_efficiency(...)`：执行实验 2A，在各真实分类任务上记录收敛轨迹，统计达到阈值准确率（如 85%）所需的最小 Epochs、时间和物理 Optical Passes；
  - `run_level3_hardware_robustness(...)`：执行实验 3A（热串扰保持率）、3B（ADC 步长网格扫描）与 3C（级联插损）。

### 文件 4：`benchmarks/plot_benchmark_results.py`
- 读取 `benchmarks/results/*.json`，生成 4 张专业学术图表（PNG 格式，保存至 `benchmarks/results/plots/`）：
  1. `fig1_scalability_and_fidelity.png`：单步耗时随 $N$ 扩展对数曲线 + Bernoulli 梯度余弦相似度曲线；
  2. `fig2_three_axis_convergence.png`：分类准确率收敛三坐标对比（Accuracy vs. Epochs, Time, Optical Passes）；
  3. `fig3_thermal_crosstalk_robustness.png`：热串扰强度 $\alpha$ 下三种范式的准确率退化曲线；
  4. `fig4_adc_quantization_heatmap.png`：ADC 比特与微扰步长的二维分类准确率热力图。

### 文件 5：`tests/test_benchmarks.py`
- 追加针对新 Benchmark 体系的单元测试：
  - `test_training_benchmark_metrics_and_adapters()`：断言 `AutogradDifferentiator` 精度对齐、链式法则 `NoisyParameterShift` 梯度方向有效、`OpticalPassTracker` 计数精确；
  - `test_training_benchmark_quick_end_to_end(tmp_path)`：调用 `--quick` 全流程，断言字典非空、结果非 NaN 且 JSON 文件正确生成。

---

## 八、四阶段实施流程与即时验证节点规范 (Execution Workflow & Verification Milestones)

为了保证代码演进严谨可控、问题就地发现与排查，执行过程严格按照自底向上的四个阶段进行，每个阶段均配备可直接执行的验证命令与硬性验收标准：

### 阶段 1：底层基础设施构建与数学校验 (Phase 1: Foundations)
- **执行内容**：
  1. 编写 `benchmarks/metrics.py`（求导适配器、测量计数器、梯度保真度度量）；
  2. 编写 `benchmarks/tasks.py`（`TaskBundle`、4 项标准分类任务加载器、双层分类模型工厂）。
- **即时验证命令**：
  ```bash
  pixi run python -c "
  import torch
  from benchmarks.tasks import get_iris_task, get_xor_task, get_digits8_task, get_digits16_task, build_optical_classifier, build_method_suite
  from benchmarks.metrics import AutogradDifferentiator, compute_gradient_metrics, OpticalPassTracker

  # 1. 验证任务集生成与非负光功率约束
  for name, getter in [('iris', get_iris_task), ('xor', get_xor_task), ('digits8', get_digits8_task), ('digits16', get_digits16_task)]:
      bundle = getter()
      assert bundle.X_train.min() >= 0.0, f'{name} X_train has negative optical power'
      assert bundle.X_train.shape[1] == bundle.input_dim, f'{name} port dimension mismatch'
      print(f'Task {name} OK: ports={bundle.input_dim}, classes={bundle.num_classes}, train={len(bundle.X_train)}')

  # 2. 验证模型前向与适配器求导对齐
  model = build_optical_classifier(num_ports=4, num_classes=3, depth=2)
  loss_fn = torch.nn.CrossEntropyLoss()
  diff = AutogradDifferentiator()
  x = torch.rand(4, 4)
  y = torch.tensor([0, 1, 2, 0])
  grad = diff.differentiate(model, x, loss_fn, y_label=y)
  assert torch.isfinite(grad).all(), 'Gradients contain NaN/Inf'
  print('Phase 1 Foundations verification PASSED!')
  "
  ```
- **验收标准**：
  * 4 类任务数据集全部在本地生成成功（零网络请求），特征非负且维度精确匹配；
  * `AutogradDifferentiator` 梯度计算成功，无任何 NaN/Inf，反向传播与前向通道顺畅。

---

### 阶段 2：三级综合基准主程序与 CLI 构建 (Phase 2: Benchmark CLI & Workflows)
- **执行内容**：
  1. 编写 `benchmarks/benchmark_training_methods.py`；
  2. 串联 Level 1（算子耗时/保真度）、Level 2（端到端收敛与三坐标数据采集）与 Level 3（热串扰/ADC量化死区压力测试）；
  3. 提供统一 CLI 参数解析器与格式化控制台输出表格。
- **即时验证命令**：
  ```bash
  # 运行快速冒烟模式验证全流程连通性（必须在 15 秒内完成）
  pixi run python benchmarks/benchmark_training_methods.py --level all --quick --output-dir benchmarks/results
  ```
- **验收标准**：
  * 终端打印格式整齐的耗时与收敛结果对比表格；
  * `benchmarks/results/` 目录下成功生成并落盘 3 份合法 JSON 文件：
    - `level1_scalability_fidelity.json`
    - `level2_classification_convergence.json`
    - `level3_hardware_robustness.json`
  * JSON 中无 `null`、`NaN` 或未捕获异常。

---

### 阶段 3：学术级结果可视化自动化流水线 (Phase 3: Automated Visualization)
- **执行内容**：
  1. 编写 `benchmarks/plot_benchmark_results.py`；
  2. 读取阶段 2 输出的 JSON 文件，绘制 4 张核心学术图表。
- **即时验证命令**：
  ```bash
  pixi run python benchmarks/plot_benchmark_results.py --input-dir benchmarks/results --output-dir benchmarks/results/plots
  ```
- **验收标准**：
  * 在 `benchmarks/results/plots/` 目录下成功输出 4 张高分辨率 PNG 图片：
    - `fig1_scalability_and_fidelity.png`
    - `fig2_three_axis_convergence.png`
    - `fig3_thermal_crosstalk_robustness.png`
    - `fig4_adc_quantization_heatmap.png`
  * 图表字号清晰、图例完整、坐标轴标注规范、无文本遮挡与布局溢出。

---

### 阶段 4：自动化回归测试加固与全库验证 (Phase 4: Regression Tests & CI Integration)
- **执行内容**：
  1. 在 `tests/test_benchmarks.py` 追加适配器、计数器与 `--quick` 流程测试用例；
  2. 执行全库 pytest 检验，确保本次新增代码零破坏、全兼容。
- **即时验证命令**：
  ```bash
  # 1. 专门运行基准测试模块
  pixi run pytest tests/test_benchmarks.py -v
  # 2. 运行项目全量测试套件
  pixi run pytest
  ```
- **验收标准**：
  * `tests/test_benchmarks.py` 所有新增用例通过；
  * 全库 220+ 项既有测试 100% 保持绿色，且无新增严重 Warning。

