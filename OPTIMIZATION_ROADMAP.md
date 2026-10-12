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

### 阶段一：硬件仿真正确性与鲁棒性修复（最高优先级 P0）
- [ ] **修复激光器相位与负值安全**：
  - 保留原始光场相位，相位噪声作为增量叠加；
  - 增加 `torch.clamp(input_power, min=0.0)`，彻底杜绝 `NaN`。
- [ ] **修复散粒噪声通道独立计算**：
  - 去除 `photocurrent.mean()`，改为逐通道独立方差 $\text{Var} = 2 q I \Delta f$。
- [ ] **严谨的 Domain 判断与功率转换保护**：
  - 重构 `AUTO` 域推断逻辑，基于计算图末端模块类型判断；
  - 转换前做数值安全检查；
  - 禁止复数张量直接流入光探测与后端电路，杜绝 ADC 中 `torch.clamp` 的 `NotImplementedError` 崩溃。
- [ ] **规范光电链路量纲与修正器件默认参数**：
  - 明确规范光场输入功率基准量纲（mW vs SI 制 W），确保光电流计算与 SI 制物理常数（$q, k_B$）自洽；
  - 修正 ADC 异常默认量程（$60\text{ V} \to 2.0\text{ V}$），匹配合理的 TIA 增益，避免 ADC 步长过大将物理散粒噪声/热噪声全部截断抹杀为 0，防止小信号截断为死神经元。

### 阶段二：测试套件深度加固与 Benchmark 基准建立（关键保障 P0）
- [ ] **铲除测试中的掩耳盗铃跳过（`pytest.skip`）**：
  - 移除 `test_differentiator.py` 中的 `pytest.skip`，彻底修复仿真器在复数输入下的异常；
- [ ] **强化求导算法数值硬断言（Hard Assertion）**：
  - 在 `test_backward_functions.py` 中，对比 `AdjointVariant`、`ParameterShift` 与原生 `Default`（Autograd）算出的梯度数值误差（`torch.testing.assert_close`），杜绝只查 shape 的假测试；
- [ ] **补齐物理酉性与能量守恒测试**：
  - 为 `MZI` 与 `ClementsMesh` 增加无损理想状态下的传递矩阵酉性校验（$U^\dagger U = I$）；
- [ ] **引入 PyTorch `gradcheck` 梯度金标准检验**：
  - 对 `CustomForwardFunction` 使用 `torch.autograd.gradcheck` 做有限差分检验；
- [ ] **构建 Benchmark 性能测试脚本**：
  - 针对不同端口尺度（$N=4, 8, 16, 32, 64$）测试前向耗时、反向耗时与显存开销，建立优化前的性能 Baseline。

### 阶段三：算力密集路径向量化与张量化重构（近期重点 P1）
- [ ] **MZI 与列矩阵批处理**：
  - 将同列所有 MZI 的 $\theta$ 和 $\phi$ 聚合为向量计算，避免逐个组件创建 $2\times 2$ 小矩阵；
  - 利用块对角稀疏矩阵或一维索引重排，一步向量化填充整列转移矩阵。
- [ ] **前向传递模式支持**：
  - 提供两种前向模式：
    1. `Matrix-based Forward`（适合超大 Batch 或导出固定矩阵，带缓存）；
    2. `Field-propagation Forward`（光场逐列传递，避免显式 $N^3$ 矩阵连乘）。
- [ ] **Adjoint 伴随法向量化**：
  - 将单列内所有 MZI 的梯度求导向量化展开，一次性利用点积完成整列梯度的计算。
- [ ] **Monte Carlo 仿真批处理并行化**：
  - 重构 `run_mc`，利用 PyTorch Batch 维度并行完成多重随机样本前向计算。

### 阶段四：内存管理、缓存机制与架构重构（P2）
- [ ] **静态缓存机制（Transfer Matrix Cache）**：
  - 在 `CoreLayer` 引入 `_cached_transfer_matrix`。当参数未发生梯度更新时直接复用缓存。
- [ ] **组件与仿真器职责统一**：
  - 将 TIA/ADC 的非理想性计算收敛到 `components.py` 中，消除冗余代码。
- [ ] **分布式级联光损耗模型**：
  - 实现逐层注入插损与噪声的机制，保证多层非线性网络的仿真保真度。
- [ ] **设备与精度规范化**：
  - 统一输入与内部张量的 `device` 和 `dtype` 处理，消除内层循环中冗余的 `.to()` 调用。

### 阶段五：编译器适配与前沿硬件特性（P3）
- [ ] **消除 Inplace 切片写入与支持 `torch.compile`**：
  - 改用函数式拼接代替切片索引赋值，测试并确保能够通过 `torch.compile(mode="reduce-overhead")` 编译。
- [ ] **热串扰矩阵（Thermal Crosstalk Matrix）建模**：
  - 引入空间相邻相移器间的热扩散串扰耦合矩阵 $C \cdot \boldsymbol{\theta}$。

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


