# ioptics

基于 PyTorch 的集成光神经网络（Integrated Optical Neural Networks, IONN）仿真、训练与物理可微计算框架，用于模拟片上硅基光子芯片的计算行为与非理想性物理效应。

## 环境管理 (Pixi)

本项目统一使用 **pixi** 进行依赖管理和任务执行：

- 安装并更新依赖：`pixi install`
- 运行测试：`pixi run test` （或 `pixi run pytest tests/`）
- 启动 Python 交互环境：`pixi run python`
- 激活开发环境 Shell：`pixi shell`
- 启动 JupyterLab：`pixi run jupyter lab`

## 仓库结构与核心分层架构

项目代码采用清晰的五层架构设计：

1. **光学器件层 (`ioptics/components.py`)**
   - 核心器件模型：单模相移器（`PhaseShifter`）、50:50分束器（`Beamsplitter`）、马赫-曾德尔干涉仪（`MZI`）、调制器（`MZM`）。
   - 光电接口器件：连续激光源（`Laser`）、光电二极管探测器（`Detector`）、跨阻放大器（`TIA`）、模数转换器（`ADC`）。
   - 定义 $2\times 2$ 复数传递矩阵、光电转换灵敏度与微观噪声方程。

2. **拓扑与网格层 (`ioptics/layers.py`)**
   - 网格拓扑：Clements 网格（`ClementsMesh`）与全连接阵列（`FCArray`）。
   - 通过多列元器件（`MeshColumn`）的传递矩阵连乘或逐列场传播构成整体 $N\times N$ 线性光学酉变换。

3. **模型顶层与非线性激活 (`ioptics/models.py`, `ioptics/nonlinear.py`, `ioptics/training.py`)**
   - 容器与训练：`OpticalNN` 顶层网络容器，统一提供 `fit(..., simulator=...)` 训练入口。
   - 光学非线性：全光自相位调制（`SPMActivation`）与电光非线性激活（`ElectroOpticActivation`）。

4. **求导算法与物理梯度策略 (`ioptics/backward_functions.py`, `ioptics/differentiators.py`)**
   - `Default`：基于 PyTorch 原生 Autograd 计算理论梯度。
   - `ParameterShift`：基于 $\pm \pi/2$ 物理相移差分的参数位移法（适合片上真实量测）。
   - `AdjointVariant`：伴随场法，利用正向光场与伴随光场逐层干涉回传，计算复杂度降至 $O(1)$。
   - `BernoulliDifferentiator` / `PhysicalBernoulliDifferentiator`：基于随机扰动的黑盒摄动优化（SPSA 思想）。

5. **硬件非理想性仿真引擎 (`ioptics/simulator.py`)**
   - `Simulator`：统一调度系统级物理效应，包括器件插入损耗（Insertion Loss）、激光器相位噪声与相对强度噪声（RIN）、光电探测器散粒噪声与热噪声、TIA 饱和与增益误差、ADC 有限精度量化与非线性（INL/DNL）。
   - 核心枚举：`HardwareProfile`（硬件规格模板）、`ExecutionMode`（执行模式：DETERMINISTIC / STOCHASTIC / STATISTICAL）、`OutputDomain`（输出域：FIELD / POWER / DETECTED_CURRENT / ADC_DIGITAL / AUTO）。

## 优化路线图进展与规划 (基于 OPTIMIZATION_ROADMAP.md)

遵循 **“先正确、强防护、再度量、后加速”** 软件工程演进准则：

### 阶段一：硬件仿真正确性与鲁棒性修复 (P0) [已完成]
- **激光器相位与安全**：保留原始光场相位 $\angle E$，增量叠加随机相位噪声；增加 `torch.clamp(min=0.0)` 防负功率开方导致 NaN。
- **散粒噪声通道独立**：废除全局 `photocurrent.mean()`，严格遵循泊松散粒噪声物理规律，按通道独立计算方差 $\text{Var} = 2 q I \Delta f$。
- **OutputDomain 严谨推断与复数保护**：基于计算图安全推断电场/功率域，严禁复数流入探测与电子器件，彻底解决 ADC 中 `torch.clamp` 的 `NotImplementedError` 崩溃。
- **规范光电链路物理量纲与参数**：规范光场输入功率基准与 SI 制物理常数自洽；将 ADC 异常默认高压量程（60V -> 2.0V）修正为芯片级标准，防止量化步长过大将物理噪声全部截断抹杀为 0 及造成小信号死神经元。
- **非负物理量分级防御**：外部输入负光功率严格抛错（`ValueError`）；激光器高斯小信号 RIN 越界触发告警（`UserWarning`）；微小舍入误差由浮点容差截断。
- **Auto-Gain 与端到端归一化**：支持自适应 TIA 跨阻阻抗调节以匹配 ADC 量程，支持 ADC 输出归一化至 $[0, 1]$ 无量纲区间。
- **API 规范化**：全库统一 PascalCase `Simulator`，重命名 `HardwareProfile` / `ExecutionMode` / `OutputDomain` 枚举，统一 `OpticalNN.fit(..., simulator=...)`。

### 阶段二：测试套件深度加固与 Benchmark 基准建立 (P0) [已完成]
- **数值硬断言（Hard Assertion）**：在测试中对比 `AdjointVariant`、`ParameterShift` 与原生 `Default`（Autograd）算出的梯度数值误差（`torch.testing.assert_close`），实现实数与复数输入下的高精度对比验证。
- **物理守恒性检验**：补齐 `MZI` 与 `ClementsMesh` 理想状态下的传递矩阵酉性校验（$U^\dagger U = I$）与 Parseval 能量守恒测试。
- **PyTorch `gradcheck` 梯度验证**：对自定义反向传播算子使用 `torch.autograd.gradcheck` 进行双精度有限差分检验；修复了 `ParameterShift` 中 inplace 修改 saved tensor 的隐藏缺陷。
- **性能基准建立**：构建 `benchmarks/benchmark_baseline.py`，完整测试不同网格规模（$N=4, 8, 16, 32, 64$）下的前向与反向耗时基线，输出 `benchmarks/baseline_results.json`，为阶段三向量化加速确立了量化标尺。

### 阶段三：核心计算路径向量化与张量化重构 (P1) [已完成]
- **MZI / MZM 与列矩阵批处理**：消除 `MZI`、`MeshColumn`、`ArrayColumn` 与 `ArrayCore`（`FCArray`）中逐器件 Python `for` 循环与张量切片就地赋值；将 `MZM` 矩阵乘法链化简为闭式解析解 $\frac{1}{2}(e^{i\theta} + 1)$，大尺寸转移矩阵构建速度提升高达 **12.3x**。
- **前向传递模式优化**：支持 `Field-propagation Forward`（光场向量逐列传递，复杂度 $O(L \cdot B \cdot N^2)$），在小 Batch 输入时避免显式 $O(L \cdot N^3)$ 全局矩阵链乘，前向计算加速高达 **16.2x**。
- **伴随法向量化**：将单列所有 `MZI` / `PhaseShifter` 及整层 `FCArray`（`MZM`）的梯度求导通过点积与矩阵外积向量化展开，代数消解 $2\times 2$ 碎小矩阵与 `F.linear` 调用，反向梯度计算加速高达 **7.3x ~ 10.6x**。
- **蒙特卡洛并行化**：利用 PyTorch Batch 广播维度将 `run_mc` 的串行循环重构为多样本批量并行执行，单次 100 次蒙特卡洛采样耗时从 5.1s 骤降至 45ms（加速 **114x**）。

### 阶段四：缓存机制与架构收敛 (P2)
- **静态转移矩阵缓存** [已完成]：构建 `OpticalParameter` 修改追踪、`ColumnLayer` 列级缓存与 `CoreLayer` 整层缓存（含 `MeshCore` 单列微扰前后缀矩阵乘积快速路径）；在 `eval()` 静态推理下 $N=64$ 延迟降至 **~0.03ms**（提速 **1500x+**），同时完整保障有限差分与参数位移等非反向传播单参数微扰训练的精确性并使 `ParameterShift` 提速 **3.8x ~ 8.3x**。
- **职责下沉** [已完成]：将 TIA（增益误差、失调、噪声、饱和截断、Auto-Gain）与 ADC（增益误差、失调、INL 三次非线性、DNL 过渡噪声）的非理想性计算统一收敛至 `components.py` 的器件类本身，消除 `simulator.py` 中的重复逻辑。
- **分布式级联光损耗** [待完成]：逐层注入插损与噪声，保证多层非线性网络免受虚高光功率失真影响。
- **规范张量生命周期与清理遗留接口** [已完成]：移除已弃用的 `OpticalNN.fit_simulation`；统一 `device` 与 `dtype` 处理，消除内层循环中冗余的 `.to()` 调用。

### 阶段五：未来极致加速与前沿硬件特性 (P3/Future)
- **拓扑级稀疏波导原位传递** [待完成]：单列仅原位更新 $N/2$ 对有效 MZI 波导端口，整网前向复杂度从 $O(N^3)$ 降至 $O(N^2)$，理论 FLOPs 再降 10x ~ 30x。
- **参数位移法批处理并行化** [已完成]：在 `ParameterShiftStrategy`、`MeshCore`/`ArrayCore.forward_parameter_shifts` 与 `ParameterShiftDifferentiator` 中将全部参数的 $\pm\pi/2$ 扰动状态在参数批维度一次性向量化并行展开，消除逐参数串行循环；$N=8$ 耗时从 `403.2ms` 降至 **`7.56ms`**（提速 **53.3x**），$N=16$ 降至 **`22.07ms`**（提速 **59.3x**），$N=64$（4032 参数）仅需 **`394.05ms`**。
- **消除 CPU-GPU 同步屏障** [已完成]：移除 Auto-Gain 与训练批循环热路径中的 `.item()` 调用，改为纯张量运算保持显存流水线。
- **消除 Inplace 写入并适配 `torch.compile`** [已完成]：将转移矩阵构建中的切片原地写入改为函数式 `scatter` 算子，支持 Dynamo 图捕获与算子融合。
- **热串扰矩阵（Thermal Crosstalk Matrix）** [待完成]：引入相邻相移器间的空间热扩散耦合矩阵。
