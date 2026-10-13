# ioptics 源代码与性能优化路线图 (Optimization Roadmap)

本文档整理了 `ioptics` 项目的核心架构分析、当前代码瓶颈及后续性能与工程优化的具体改进方案，供后续优化迭代参考。

---

## 一、项目架构与数据流速览

`ioptics` 是基于 **PyTorch** 的集成光神经网络（IONN）仿真与训练框架：

1. **元件层 (`ioptics/components.py`)**：单模相移器（PhaseShifter）、50:50分束器（Beamsplitter）、马赫-曾德尔干涉仪（MZI）、调制器（MZM）等，定义 $2\times 2$ 复数转移矩阵及相位噪声模型。
2. **阵列拓扑层 (`ioptics/layers.py`)**：Clements 网格（`ClementsMesh`）与全连接阵列（`FCArray`）。多列元器件通过矩阵连乘 $T = \prod_{i=L}^1 T_i$ 构成整网线性光学传递矩阵。
3. **求导算法 (`ioptics/backward_functions.py`, `differentiators.py`)**：
   - 自动微分：`Default`（PyTorch Autograd）
   - 物理测量导向：`ParameterShift`（$\pm \pi/2$ 相移差分）
   - 快速伴随法：`AdjointVariant`（正向场与伴随场逐层回传，梯度复杂度降至 $O(1)$）
   - 随机黑盒摄动：`BernoulliDifferentiator` / `PhysicalBernoulliDifferentiator`（SPSA 思想）
4. **非理想性仿真 (`ioptics/simulator.py`)**：系统级硬件效应（插损、探测噪声、RIN、TIA 饱和、ADC 量化）。
5. **模型与训练 (`ioptics/models.py`, `training.py`, `nonlinear.py`)**：`OpticalNN` 统一容器与训练控制器。

---

## 二、当前核心性能瓶颈分析

### 1. 转移矩阵构建缺乏向量化（高优先级瓶颈）
- **问题现状**：
  - 在 `MeshColumn.get_transfer_matrix` 中，采用 Python 逐组件 `for component in self.components` 循环遍历。
  - 在大矩阵上进行切片就地写入（`transfer_matrix[m][m] = T[0,0]`, `transfer_matrix[m][n] = T[0,1]` 等）。
  - 在 `MZI.get_transfer_matrix` 中，每个元器件每次调用都会创建一个独立的 `torch.zeros(2, 2)` 并逐元素索引赋值。
  - **影响**：当网格端口数 $N$ 增大（例如 $N=16, 32, 64$ 时，元器件数约为 $N^2/2$），构建整网转移矩阵会产生大量的 Python 解释器开销与碎小张量分配。

### 2. 矩阵链乘开销与计算复杂度问题
- **问题现状**：
  - `MeshCore.get_transfer_matrix` 中使用 `reduce(torch.matmul, [layer.get_transfer_matrix() for layer in reversed(self.columnlayers)])`。
  - 生成 $L$ 个 $N \times N$ 矩阵并连续相乘，矩阵-矩阵乘法的总体复杂度为 $O(L \cdot N^3)$。
- **潜在优化**：
  - 对于小 Batch 输入（$B \ll N$），直接将光场向量逐列传递计算（$x_{k+1} = x_k T_k$），每次是向量-矩阵乘法，总体复杂度为 $O(L \cdot B \cdot N^2)$，比显式生成全局转移矩阵更经济。
  - 静态/评估模式下，若权重未更新，可增加缓存机制（Transfer Matrix Cache），避免每次 forward 都重新计算完整拓扑。

### 3. 伴随法（AdjointStrategy）中的频繁小张量分配
- **问题现状**：
  - 在 `AdjointStrategy.compute_gradients` 的反向传播循环中，每个 MZI 都临时构造 `dT_dtheta = torch.zeros(2, 2)` 和 `dT_dphi = torch.zeros(2, 2)`，并逐个调用 `F.linear`。
  - 产生大量临时对象创建、析构与内存分配器碎片，严重拖慢 GPU/CPU 执行速度。

### 4. 冗余的类型与设备同步开销
- **问题现状**：
  - 核心循环与算子内部频繁出现 `.to(device=..., dtype=...)`（例如 `to(device, dtype=internal_dtype)`）。
  - 如果未统一管理张量生命周期，会引入额外的同步屏障和数据搬移开销。

### 5. 对现代 PyTorch 特性（`torch.compile`）支持不足
- **问题现状**：
  - 存在大量基于 Python 列表、切片写入和动态控制流的操作，难以被 TorchDynamo 和 Inductor 后端捕获，阻碍了算子融合（Operator Fusion）优化。

---

## 三、硬件仿真模块专项问题分析 (Hardware Simulator)

硬件仿真模块（`ioptics/simulator.py` 与 `ioptics/components.py`）目前存在多项物理建模失真、数学计算缺陷与性能瓶颈：

### 1. 激光器仿真将原始光场相位暴力擦除（严重物理失真 + NaN 隐患）
- **位置**：`components.py: Laser.simulate`
- **问题现状**：
  - 原始光场 $E$ 取模计算功率后，直接丢弃原相位，将其覆盖为由激光线宽随机生成的噪声相位 `phase`，破坏了输入复数场的相干性与相位编码。
  - `input_power + self.RIN_noise(...)` 发生负值时（低光强下可能出现），紧接着执行 `torch.sqrt(...)` 会直接导致 `NaN` 产生。
- **改进建议**：
  - 提取原光场相位 $\angle E$，并将随机相位抖动作为增量叠加：`phase_total = torch.angle(E) + phase_noise`。
  - 在计算功率和振幅开方前增加 `torch.clamp(min=0.0)` 防负值。

### 2. 探测器散粒噪声方差计算使用全局 `mean()`（物理建模错误）
- **位置**：`components.py: Detector.total_noise`
- **问题现状**：
  - 散粒噪声方差使用 `shot_noise_var = 2 * q * photocurrent.mean() * self.bandwidth` 计算。
  - 泊松散粒噪声物理公式为 $\text{Var}(I_{\text{shot}}) = 2 q I \Delta f$，正比于当前信道局部光电流。全局 `mean()` 导致弱光端口被强光端口人为拉高噪声，破坏了通道局部独立的信噪比（SNR）分布。
- **改进建议**：
  - 移除 `mean()`，直接按逐元素计算方差：`shot_noise_var = 2 * q * photocurrent * self.bandwidth`，各通道独立生成噪声。

### 3. 光损耗（Insertion Loss）作为后置全局衰减，忽略级联非线性效应
- **位置**：`simulator.py: simulator._run_internal`
- **问题现状**：
  - 先跑完整个无损的 `optical_model`，统计器件总数计算全局 `loss_db`，最后在输出端一次性乘以缩放因子。
  - 在包含非线性激活函数（如 `SPMActivation`, `ElectroOpticActivation`）的多层网络中，非线性层接收到的光功率严重虚高，使得非线性响应被严重夸大。
- **改进建议**：
  - 支持分布式/逐层损耗传递模型，在每级 MZI 列或核心层后实时按比例注入插损。

### 4. 架构割裂：`components.py` 与 `simulator.py` 逻辑重复
- **位置**：`components.py`（Laser, Detector, TIA, ADC）与 `simulator.py`（`_apply_tia_non_idealities`, `_apply_adc_non_idealities`）
- **问题现状**：
  - TIA 与 ADC 的非理想性（失调、增益误差、INL、DNL）只存在于 `simulator.py` 中，与 `components.py` 的类相互割裂，组件无法单独作为具有非理想性的模块使用。
- **改进建议**：
  - 将 TIA 与 ADC 的非理想参数与计算下沉封装到组件类本身，`simulator.py` 仅负责调度与参数配置注入。

### 5. 输出域（Field vs Power）自动推断规则脆弱
- **位置**：`simulator.py: simulator._run_internal`
- **问题现状**：
  - 在 `AUTO` 模式下仅凭 `torch.all(output >= 0)` 推断是否为功率。若实数电场碰巧全部为非负值，会被误判为功率域，导致衰减计算相差一倍指数（$10^{-dB/20}$ vs $10^{-dB/10}$），并误跳过探测器的模方计算。
- **改进建议**：
  - 显式通过模型各层属性（如末尾是否包含 `Abs` / `AbsSquared`）进行结构推断，或强制要求用户/模型标注 `output_domain`，并在功率转换前做有效范围校验。

### 6. 蒙特卡洛仿真（`run_mc`）采用串行 Python 循环，耗时极长
- **位置**：`simulator.py: simulator.run_mc`
- **问题现状**：
  - 执行 $N$ 次采样时，在 Python 解释器中通过 `for _ in range(n_samples)` 循环单步调用 `self.run(...)`。
- **改进建议**：
  - 引入批处理（Batch Parallelization）：将输入张量扩展为 `(n_samples, batch_size, ...)`，一次性通过光网络前向批计算，结合并行噪声张量生成，提升 20x~100x 执行效率。

### 7. 光电转换链路量纲失配与 ADC 参数异常（物理噪声被完全吞噬 + 复数崩溃 Bug）
- **位置**：`components.py: Detector, ADC, TIA` 与 `simulator.py: SimulatorConfig, _apply_adc_non_idealities`
- **问题现状**：
  1. **输入光场单位隐式假设不一致**：
     - `Detector.responsivity` 默认值为 `1.0e-3`（注释标注 `1.0 A/W (Si @ 1550nm), 1e-3 A/mW`），代码隐式假设输入功率单位为毫瓦（mW）；
     - 但散粒噪声与热噪声公式中的电子电荷 $q$ 与玻尔兹曼常数 $k_B$ 均采用标准 SI 制单位；
     - 当机器学习用户输入未归一化或量纲不明确的张量时，导致物理计算的绝对信噪比（SNR）严重失真。
  2. **ADC 量程异常（高压 60V）导致量化步长过大，完全吞噬物理噪声**：
     - `ADC` 默认 `v_range = 60.0` V，`bits = 12`，导致量化步长 $\text{LSB} = 60.0 / 4096 \approx 14.65\text{ mV}$；
     - 探测器的物理热噪声与散粒噪声在微安级，经 $100\ \Omega$ TIA 放大后电压噪声标准差 $\sigma_v \approx 0.1\text{ mV} \ll \text{LSB}$；
     - **后果**：多次随机噪声采样输出的标准差恒等于 0，精心建模的物理噪声在输出端全部被 ADC 粗暴地截断抹杀；微弱光信号（如输入功率 $< 0.001$）直接被截断为 0V，导致死神经元。
  3. **复数光场直传 ADC 导致 `torch.clamp` 抛出 `NotImplementedError`**：
     - 当模型未显式加模方探测层且 `output_domain=POWER` 时，复数张量一路传递至 ADC，在 `torch.clamp(v, 0.0, v_range)` 时直接抛出崩溃，成为此前测试不得不加 `pytest.skip` 掩盖的底层根因。
- **改进建议**：
  - **明确光场输入基准与增益匹配**：在 `SimulatorConfig` 中规范光功率量纲基准与缩放系数（或支持归一化功率映射），使光电探测电流与 SI 制物理常数严格自洽；
  - **修正 ADC 与 TIA 默认参数**：将 `adc_v_range` 修正为符合芯片级标准的数值（例如 $2.0\text{ V}$ 或 $3.3\text{ V}$），并匹配合理的 TIA 跨阻增益，确保信号动态范围与物理噪声在 ADC 满量程内合理呈现；
  - **复数保护与类型校验**：在进入光电探测（Detector/TIA/ADC）前强制进行光功率实数转换（取模方或报错提醒），严禁复数张量流入后端电子器件模拟。

### 8. 非负物理量“静默截断（Silent Clamp）”弊端与分级防御机制
- **位置**：`components.py: Laser.simulate, Detector.simulate` 与 `simulator.py: Simulator._run_internal`
- **问题现状**：
  - 代码中对光功率非负要求常使用 `torch.clamp(min=0.0)` 兜底。单纯的静默截断虽然能消除 NaN 崩溃，但带来两项严重隐患：
    1. **掩盖输入错误**：若用户传入的光功率本身含有显著负值（如漏接探测层，或电场实数负值误当功率），静默截断会导致大片神经元输出恒为 0 且梯度消失，用户难以察觉；
    2. **破坏物理噪声均值统计**：高斯分布的 RIN 相对强度噪声是小信号线性近似（$\delta P \ll P_0$）。当光强极弱或噪声极大时，高斯负尾巴越过 0 点，单纯截断至 0 会导致平均功率被非物理地人为拉高，破坏零均值噪声特性。
- **分级防御策略**：
  - **强前置负值校验（Strict Validation $\to$ `ValueError`）**：针对用户/模型直接作为“光功率”传入的张量，若检测到显著负值（$< -1e-6$），立即报错拒绝，提示检查模型结构或输入域配置；
  - **高斯噪声物理越界告警（Physics Breakdown Warning $\to$ `UserWarning`）**：在激光器内部叠加 RIN 噪声后，若出现显著负值，触发告警提示高斯小信号近似在当前弱光/大噪声工作点已失效，同时截断为 0 保证计算不挂断；
  - **微小浮点容差（Numerical Tolerance）**：$[-1e-6, 0)$ 范围内的微小负值视作浮点舍入误差，静默对齐至 0。

---

## 四、实施顺序规划与因果逻辑 (Execution Ordering & Strategy)

遵循科学的 **“先正确、强防护、再度量、后加速”** 软件工程法则，各阶段的实施顺序与依赖逻辑如下：

```
第 1 步：修复物理与数值正确性 (P0) ──► 确保物理真实性、杜绝 NaN 崩溃
          │
第 2 步：测试加固与 Benchmark 基准 (P0) ──► 补齐真实梯度断言与物理守恒，立性能标尺（装好心电监护仪）
          │
第 3 步：核心计算路径向量化 (P1)  ──► 消除 Python for 循环与切片写入，实现成倍加速
          │
第 4 步：蒙特卡洛并行化 (P1)      ──► 利用 PyTorch Batch 维度将 100 次串行重构为批量并行
          │
第 5 步：架构收敛与高级物理特性 (P2/P3) ──► 级联损耗、缓存机制、热串扰矩阵、torch.compile
```

### 为什么按这个顺序推进？
1. **为什么第 1 步必须修物理与数值正确性 (P0)？**
   - **“跑得快的前提是算得对”**。激光器抹除相位会导致相干光特征丢失，负值开方会导致训练在半途中 `NaN` 崩溃；散粒噪声误用 `mean()` 破坏了通道信噪比。
   - 如果不先修复底层物理与数值计算，在此基础上进行的复杂张量广播重构可能会加速一个“错误且脆弱”的算法，后续再去调整物理公式将大幅增加返工成本。
2. **为什么在重构向量化前必须先“加固测试 + 建立 Benchmark”？**
   - **“动大手术之前，必须先装上高精度心电监护仪”**：
     - 当前测试存在严重的“弱断言”（只查梯度形状，不查导数数值是否正确），甚至存在用 `pytest.skip` 掩盖复数崩溃的现象；
     - 如果在弱断言的基础上去大改底层张量算法，改错梯度数值时测试依然会“假绿灯”；
     - 先把测试断言加固成**硬数值对比（Adjoint 必须精确等于 Autograd 理论导数，网格必须满足酉矩阵能量守恒）**，后续做向量化大重构时，测试套件就能提供 100% 可信的安全网兜底。
   - **“没有度量就没有优化”**：
     - 记录 $N=4, 8, 16, 32, 64$ 下的前向与反向耗时基线，优化后能立即用客观数据量化加速比。
3. **为什么核心向量化排在蒙特卡洛前面？**
   - 蒙特卡洛多样本批处理依赖于网络具备良好的 Batch 扩展能力。先完成了 MZI 与网格层的向量化张量重构，蒙特卡洛的 Batch 并行化才能自然平滑地落地。
4. **为什么架构收敛和高级特性放最后？**
   - 级联损耗模型、热串扰矩阵与 `torch.compile` 属于进阶物理保真度与编译器融合特性，需要建立在底层稳固、高速的核心算力架构之上。

---

## 五、优化推进任务清单 (Task Checklist)

### 阶段一：硬件仿真正确性与鲁棒性修复（最高优先级 P0）(Completed ✅)
- [x] **修复激光器相位与负值安全**：
  - 保留原始光场相位，相位噪声作为增量叠加；
  - 增加 `torch.clamp(input_power, min=0.0)`，彻底杜绝 `NaN`。
- [x] **修复散粒噪声通道独立计算**：
  - 去除 `photocurrent.mean()`，改为逐通道独立方差 $\text{Var} = 2 q I \Delta f$。
- [x] **严谨的 Domain 判断与功率转换保护**：
  - 重构 `AUTO` 域推断逻辑，基于计算图末端模块类型判断；
  - 转换前做数值安全检查；
  - 禁止复数张量直接流入光探测与后端电路，杜绝 ADC 中 `torch.clamp` 的 `NotImplementedError` 崩溃。
- [x] **规范光电链路量纲与修正器件默认参数**：
  - 明确规范光场输入功率基准量纲（mW vs SI 制 W），确保光电流计算与 SI 制物理常数（$q, k_B$）自洽；
  - 修正 ADC 异常默认量程（$60\text{ V} \to 2.0\text{ V}$），匹配合理的 TIA 增益，避免 ADC 步长过大将物理散粒噪声/热噪声全部截断抹杀为 0，防止小信号截断为死神经元。
- [x] **建立非负物理量分级校验与越界告警机制**：
  - 针对外部输入的负光功率抛出 `ValueError` 阻断错误传播；
  - 针对激光器高斯小信号 RIN 噪声导致的越界负值触发 `UserWarning` 告警；
  - 对 $[-1e-6, 0)$ 微小数值提供浮点容差截断。
- [x] **实现模拟域 Auto-Gain TIA 与数字域端到端归一化**：
  - 模拟域：提供 `auto_gain=True` 选项，根据峰值光电流动态调节跨阻阻抗，使电压信号刚好达到 $\sim 85\%\ V_{\text{range}}$，消除有效位损失并最大化 ADC 采样精度；
  - 数字域：提供 `normalize=True` 选项，对 ADC 输出做标定除法映射到 $[0, 1]$ 无量纲区间，让上层深度学习算法和 Loss 计算免受绝对电压物理量纲干扰。

### 阶段二：测试套件深度加固与 Benchmark 基准建立（关键保障 P0）(Completed ✅)
- [x] **铲除测试中的掩耳盗铃跳过（`pytest.skip`）与复数输入缺陷修复**：
  - 彻底清理吞异常的不当 skip，新增 `PhysicalBernoulli` 与 `ParameterShift` 在复数输入及 `Simulator` 介入下的端到端求导测试。
- [x] **强化求导算法数值硬断言（Hard Assertion）**：
  - 在 `test_backward_functions.py` 中，对比 `AdjointVariant`、`ParameterShift` 与原生 `Default`（Autograd）算出的梯度数值误差（`torch.testing.assert_close`），实现实数与复数输入下的高精度对比验证。
- [x] **补齐物理酉性与能量守恒测试**：
  - 为 `MZI` 与 `ClementsMesh`（不同尺度 $L=2, 3, 4, 6, 8$）补齐无损理想状态下的传递矩阵酉性校验（$U^\dagger U = I$）与 Parseval 能量守恒测试。
- [x] **引入 PyTorch `gradcheck` 梯度金标准检验**：
  - 对 `CustomForwardFunction` 的 `ADJOINT_VARIANT` 与 `PARAMETER_SHIFT` 使用 `torch.autograd.gradcheck` 进行双精度有限差分检验；修复了 `ParameterShift` 中 inplace 修改 saved tensor 的隐藏 Bug。
- [x] **构建 Benchmark 性能测试脚本与 Baseline 数据建立**：
  - 构建 `benchmarks/benchmark_baseline.py` 与自动化冒烟测试；测试不同端口尺度（$N=4, 8, 16, 32, 64$）下的矩阵构建、前向与反向耗时，生成 `benchmarks/baseline_results.json` 性能标尺：

| 规模 (N) | TM Build (ms) | Fwd (B=1) (ms) | Fwd (B=32) (ms) | Bwd (Autograd) (ms) | Bwd (Adjoint) (ms) | Bwd (ParamShift) (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **4** | 2.435 | 2.354 | 2.574 | 6.588 | 8.646 | 25.260 |
| **8** | 12.164 | 12.320 | 12.832 | 31.140 | 40.389 | 403.154 |
| **16** | 46.798 | 47.693 | 47.184 | 125.257 | 166.014 | N/A (跳过超大) |
| **32** | 214.706 | 219.477 | 214.516 | 548.537 | 717.119 | N/A (跳过超大) |
| **64** | 899.530 | 940.155 | 854.596 | 2307.264 | 3035.021 | N/A (跳过超大) |

### 阶段三：算力密集路径向量化与张量化重构（P1）(Completed ✅)
- [x] **MZI / MZM 与列矩阵批处理**：
  - 优化 `MZI.get_transfer_matrix`，复用指数运算并消除就地切片赋值；将 `MZM.get_transfer_matrix` 的 3 个 $2\times 2$ 矩阵连乘化简为闭式解析解 $\frac{1}{2}(e^{i\theta} + 1)$；
  - 重构 `MeshColumn`、`ArrayColumn` 与 `ArrayCore`（`FCArray`），预计算端口索引并一次性向量化批量构建整列/整层转移矩阵。
- [x] **前向传递模式支持 (Field-propagation)**：
  - 引入 `ForwardMode`（`MATRIX` 与 `FIELD_PROP`），支持光场逐列传递，复杂度降至 $O(L \cdot B \cdot N^2)$；
  - 补齐等价性与梯度校验单元测试。
- [x] **Adjoint 伴随法向量化**：
  - 代数展开 $\partial T / \partial \theta$ 与 $\partial T / \partial \phi$ 干涉项，消除单组件 $2\times 2$ 矩阵频繁创建与 `F.linear` 调用；
  - 利用点积、张量广播与矩阵外积（`adj^H @ x`）一次性完成整列 `MZI` / `PhaseShifter` 及整层 `FCArray`（`MZM`）伴随梯度计算。
- [x] **Monte Carlo 仿真批处理并行化**：
  - 重构 `run_mc`，利用 PyTorch Batch 广播维度实现多样本并行仿真，加速比超过 **110x**。

#### 阶段三优化加速成果对比 (Benchmark: Baseline vs. Phase 3)

| 规模 (N) | 指标 | 优化前 (Baseline) | 优化后 (Phase 3) | **加速比 (Speedup)** |
| :--- | :--- | :--- | :--- | :--- |
| **N=4** | TM Build | 2.435 ms | 1.149 ms | **2.12x** |
| | Fwd (B=1) | 2.354 ms | 1.114 ms | **2.11x** |
| | Bwd (Autograd) | 6.588 ms | 3.629 ms | **1.82x** |
| | Bwd (Adjoint) | 8.646 ms | 5.160 ms | **1.68x** |
| | Bwd (ParamShift)| 25.260 ms | 14.136 ms | **1.79x** |
| **N=8** | TM Build | 12.164 ms | 3.102 ms | **3.92x** |
| | Fwd (B=1) | 12.320 ms | 2.628 ms | **4.69x** |
| | Fwd (B=32) | 12.832 ms | 2.519 ms | **5.09x** |
| | Bwd (Autograd) | 31.140 ms | 7.807 ms | **3.99x** |
| | Bwd (Adjoint) | 40.389 ms | 12.118 ms | **3.33x** |
| | Bwd (ParamShift)| 403.154 ms | 120.537 ms | **3.34x** |
| **N=16** | TM Build | 46.798 ms | 5.933 ms | **7.89x** |
| | Fwd (B=1) | 47.693 ms | 7.231 ms | **6.60x** |
| | Fwd (B=32) | 47.184 ms | 5.647 ms | **8.36x** |
| | Bwd (Autograd) | 125.257 ms | 18.612 ms | **6.73x** |
| | Bwd (Adjoint) | 166.014 ms | 32.395 ms | **5.12x** |
| **N=32** | TM Build | 214.706 ms | 18.759 ms | **11.45x** |
| | Fwd (B=1) | 219.477 ms | 24.063 ms | **9.12x** |
| | Fwd (B=32) | 214.516 ms | 18.882 ms | **11.36x** |
| | Bwd (Autograd) | 548.537 ms | 58.838 ms | **9.32x** |
| | Bwd (Adjoint) | 717.119 ms | 170.432 ms | **4.21x** |
| **N=64** | TM Build | 899.530 ms | 73.071 ms | **12.31x** 🚀 |
| | Fwd (B=1) | 940.155 ms | 57.957 ms | **16.22x** 🚀 |
| | Fwd (B=32) | 854.596 ms | 52.946 ms | **16.14x** 🚀 |
| | Bwd (Autograd) | 2307.264 ms | 218.060 ms | **10.58x** 🚀 |
| | Bwd (Adjoint) | 3035.021 ms | 417.663 ms | **7.27x** 🚀 |

### 阶段四：内存管理、缓存机制与架构收敛（P2）(Completed ✅)
- [x] **静态缓存机制（Transfer Matrix Cache）与局部微扰加速 (Completed ✅)**：
  - 引入 `OpticalParameter` 全局修改纪元追踪、二级缓存（`ColumnLayer` 列级全参数精确缓存 + `CoreLayer` 整层缓存）及 `MeshCore` 单列微扰前后缀乘积（`prefix @ T_c @ suffix`）快速路径；
  - 100% 兼容有限差分（`.data` 原地修改）、参数位移（`ParameterShift`）与多层网络局部微扰，无假命中隐患；
  - $N=64$ 评估模式（`model.eval()`）单次前向耗时降至 **~0.03ms**（提速 **1500x+**），同时单列微扰缓存复用使 $N=8$ `ParameterShift` 从 120.5ms 降至 **31.9ms**，$N=16$ 从 1309ms 降至 **158ms**（提速 **8.3x**），伴随法反向传播亦因复用列缓存进一步提速约 **1.4x**。
- [x] **组件与仿真器职责统一 (Completed ✅)**：
  - 将 TIA（增益误差、直流失调、高斯噪声、上下限饱和截断、Auto-Gain 动态跨阻调节）与 ADC（增益误差、直流失调、三次 INL 非线性失真、DNL 过渡噪声及复数保护）的非理想性参数与计算下沉封装至 `components.py` 的 `TIA` 与 `ADC` 类中；
  - 移除 `simulator.py` 中冗余的 `_apply_tia_non_idealities` 与 `_apply_adc_non_idealities` 外部函数，统一由 `TIA.simulate()` 与 `ADC.simulate()` 完成仿真。
- [x] **分布式级联光损耗模型 (Completed ✅)**：
  - 在 `CoreLayer`（`MeshCore`、`ClementsMesh`、`ArrayCore`、`FCArray`）中支持层间插损 `insertion_loss_db` 与仿真器动态覆写机制，在 `SimulatorConfig` 与 `Simulator.run` 中默认启用 `cascaded_loss=True` 逐层注入损耗（兼容 `LAYER` 与 `COMPONENT` 两种损耗模型），并在 `SimulationTrace.layer_outputs` 中记录逐层衰减后的中间光场，消除多层非线性网络（`SPMActivation`、`ElectroOpticActivation` 等）中间层光功率虚高失真。
- [x] **设备与精度规范化及遗留接口清理 (Completed ✅)**：
  - 移除已弃用的 `OpticalNN.fit_simulation` 包装接口与遗留别名；
  - 统一输入与内部张量的 `device` 和 `dtype` 处理，利用原生复数标量乘法与预注册组件 buffer 消除内层循环中冗余的 `.to()` 调用。

### 阶段五：未来极致加速与前沿硬件特性（Future 展望 / 待后续迭代）
- [ ] **拓扑级稀疏波导原位传递（Sparse Waveguide In-place Propagation）**：
  - **背景**：当前逐列场传递中每列仍构建 $N\times N$ 矩阵并执行全矩阵乘法（$O(N^2)$）。
  - **方案**：物理上每列仅有 $N/2$ 个 MZI 作用于端口 $(m_k, n_k)$，其余直通波导保持原值。通过原位仅更新 $x[..., m]$ 与 $x[..., n]$ 局部端口，单列 FLOPs 从 $O(N^2)$ 降至 $O(N)$，整网前向总复杂度从 $O(N^3)$ 降至 $O(N^2)$，运算量理论再降 10x ~ 30x。
- [x] **参数位移法（Parameter Shift）批处理并行化 (Completed ✅)**：
  - **背景**：此前片上原位调谐 `ParameterShiftStrategy` 与 `ParameterShiftDifferentiator` 采用 Python `for` 逐参数串行扰动，对于 $N \ge 16$ 耗时极长。
  - **方案**：在 `ParameterShiftStrategy`、`MeshCore`/`ArrayCore.forward_parameter_shifts` 与 `ParameterShiftDifferentiator` 中，在参数批维度一次性向量化展开全部参数的 $\pm\pi/2$ 扰动状态，消除逐参数 Python 循环与全网转移矩阵重乘；`Bwd (ParamShift)` 在 $N=8$ 从 Baseline 的 `403.15ms` 骤降至 **`7.56ms`**（提速 **53.3x**），$N=16$ 从 `1309ms` 降至 **`22.07ms`**（提速 **59.3x**），$N=32$ 仅需 **`66.59ms`**，$N=64$（4032 参数）由原先不可行的数十秒缩短至 **`394.05ms`**。
- [x] **消除 Host-Device (CPU-GPU) 同步屏障 (Completed ✅)**：
  - 消除仿真链路 Auto-Gain 中的 `torch.max(photocurrent).item()` 调用（改为纯张量 `torch.amax` + `torch.where`）及训练批循环中的逐 Batch `.item()` 同步，避免打断 GPU 异步流水线（Pipeline Stall），确保纯显存内高速流转。
- [x] **现代 PyTorch 编译器适配与算子融合（`torch.compile`）(Completed ✅)**：
  - 预计算 `MeshColumn` 稀疏散射索引 `_mzi_scatter_idx` 与基底单位阵 `_base_eye_flat`，将切片就地写入改为函数式非原地 `scatter` 算子，支持 `torch.compile` 图捕获与算子融合。
- [x] **热串扰矩阵（Thermal Crosstalk Matrix）空间物理建模 (Completed ✅)**：
  - 在 `ColumnLayer`（`MeshColumn` / `ArrayColumn`）中支持基于空间波导端口距离衰减的近邻热串扰矩阵 $C_{ij} = \alpha^{|p_i - p_j|}$ 与自定义实测矩阵注入；在有效相位 $\boldsymbol{\theta}_{\text{eff}} = C \boldsymbol{\theta} + \boldsymbol{\delta\theta}_{\text{noise}}$ 上计算中间光场与局部干涉导数，并通过链式法则 $C^\top \nabla_{\boldsymbol{\theta}_{\text{eff}}} \mathcal{L}$ 实现与 PyTorch `Autograd` 机器精度对齐；在 `SimulatorConfig` 中将 `HardwareProfile.LAB`（1%）与 `HARDWARE_LIKE`（3%）配置热串扰，并在确定性模式下自动抑制。

---

## 六、API 命名规范化重构记录 (Completed ✅)

为了消除当前 API 命名中的歧义、同名重叠、大小写反规范以及拼写错误，计划直接对全库进行标准规范重构（不保留旧别名，保持代码纯粹与现代化）。

### 1. 影响范围与工程量评估
- **全项目引用总数**：约 120 处
- **涉及文件清单**（共 7 个文件）：
  - 核心源码：`ioptics/simulator.py`、`ioptics/components.py`、`ioptics/models.py`
  - 单元测试：`tests/test_simulator_p0.py`、`tests/test_models.py`
  - 示例演示：`examples/physical_bernoulli_example.ipynb`、`examples/async_hybrid_iris_example.ipynb`
  - 项目文档：`README.md`
- **工程量评估**：轻量级重构，耗时约 5~10 分钟，依托现有 `pytest` 测试套件可实现 100% 零风险验证。

### 2. 全局重构对照表 (Renaming Mapping)

| 类别 | 现有旧命名 | 推荐标准新命名 | 重构理由与语义改进 |
| :--- | :--- | :--- | :--- |
| **类名规范** | `class simulator:` | `class Simulator:` | 符合 PEP 8 类名 PascalCase 规范；`sim = Simulator(...)` 表达清晰 |
| **硬件规格枚举** | `SimulatorProfile` | `HardwareProfile` | 明确为“器件物理规格/噪声模板” |
| **执行模式枚举** | `SimulationMode` | `ExecutionMode` | 明确为“计算执行方式（单次/随机/统计）”，彻底与 Profile 语义解耦 |
| **枚举同名消除** | `SimulationMode.IDEAL` | `ExecutionMode.DETERMINISTIC` | 消除与 Profile 中 `IDEAL` 的同名混淆，明确其含义为“无随机采样的确定性计算” |
| **输出域枚举** | `ModelOutputDomain` | `OutputDomain` | 与 `InputDomain` 形成完全对称，精简冗余的 `Model` 前缀 |
| **拼写错误纠正** | `impendance` (在 `Detector` 中) | `impedance` | 修复多出一个字母 `n` 的低级拼写 Bug，统一全库阻抗命名 |
| **网格列损耗参数** | `mzi_insertion_loss` | `layer_loss_db` | 消除与 `mzi_loss` 的歧义，明确它是“按网格列（Column）”计算的损耗 |
| **器件级损耗参数** | `mzi_loss` | `component_loss_db` | 明确它是“按独立 MZI 实例”计算的损耗 |
| **训练接口统一** | `model.fit_simulation(...)` | `model.fit(..., simulator=...)` | 统一训练主入口，避免因是否接入仿真器而产生两套孤立的训练 API |

### 3. 重构实施状态 (Status: 已完成 ✅)
1. **阶段 1：核心底层类与拼写修复 [已完成]**
   - 修复 `Detector` 中的 `impedance` 拼写；
   - 将 `class simulator:` 改为 `class Simulator:`。
2. **阶段 2：枚举与配置参数重构 [已完成]**
   - 将 `SimulatorProfile` -> `HardwareProfile`；
   - 将 `SimulationMode` -> `ExecutionMode`，并将 `.IDEAL` 更新为 `.DETERMINISTIC`；
   - 将 `ModelOutputDomain` -> `OutputDomain`，同步在 `SimulatorConfig` 中对称补齐 `input_domain`；
   - 重命名 `layer_loss_db` 与 `component_loss_db`；
   - 统一 `OpticalNN.fit(..., simulator=...)`。
3. **阶段 3：测试用例与 Examples 同步 [已完成]**
   - 同步修改 `tests/test_simulator_p0.py`、`tests/test_models.py`、`tests/test_components.py`；
   - 同步修改 `examples/` 下的所有 notebook 与 `README.md`。
4. **阶段 4：自动化回归验证 [已完成]**
   - 全库检索确认所有旧命名已全部清除。

---

## 七、单元测试套件缺陷分析与加固方案 (Test Suite Hardening)

经对 `tests/` 目录下全部测试的逐行审查，当前测试套件存在以下不合理设计与潜在隐患，需在推进核心性能向量化前予以加固：

### 1. 存在用 `pytest.skip` 掩盖底层 Bug 的现象
- **位置**：`tests/test_differentiator.py:315-318`
- **问题**：在测试物理伯努利求导时，捕获到复数输入的 `RuntimeError` 崩溃，代码直接执行 `pytest.skip(...)` 将其吞掉，导致测试看似绿灯，实则掩盖了复数处理缺陷。
- **加固方案**：彻底修复 `Simulator` 在复数输入下的异常，移除 `pytest.skip`，强制其通过断言检验。

### 2. 核心求导算法存在“弱断言”（只查形状，不查导数数值）
- **位置**：`tests/test_backward_functions.py:102-140`
- **问题**：对 `AdjointVariant` 与 `ParameterShift` 梯度的测试仅断言 `input_tensor.grad is not None` 和 `shape` 正确。即使导数数值完全算错，测试依然会通过。
- **加固方案**：引入“硬数值断言”——在相同权重下，将伴随法算出的参数梯度与 PyTorch 原生自动微分（`Default`）严格对比：
  ```python
  torch.testing.assert_close(layer_adjoint.theta.grad, layer_default.theta.grad, atol=1e-4, rtol=1e-3)
  ```

### 3. 激光器测试规避非零复数输入
- **位置**：`tests/test_components.py:73-81`
- **问题**：只测试了输入全为 0 时的输出，刻意避开了非零复数电场输入，掩盖了相位丢失和负值开方 `NaN` 的物理缺陷。
- **加固方案**：编写非零复数光场输入测试，验证相位增量叠加与负功率防 `NaN` 截断。

### 4. 物理守恒性（酉变换 $U^\dagger U = I$）校验缺失
- **位置**：`tests/test_components.py`, `tests/test_layers.py`
- **问题**：仅对分束器测了酉性，对核心器件 `MZI` 与 `ClementsMesh` 从未测试过无损理想状态下的能量守恒。
- **加固方案**：补齐 `MZI` 和 `ClementsMesh` 理想传递矩阵的 $U^\dagger U = I$ 酉性断言。

### 5. 缺失 PyTorch 官方金标准梯度检验（`gradcheck`）
- **位置**：`tests/test_backward_functions.py`
- **问题**：自定义算子 `CustomForwardFunction` 从未经过 `torch.autograd.gradcheck` 的有限差分检验。
- **加固方案**：添加基于双精度（`float64/complex128`）的 `gradcheck` 自动化测试。


