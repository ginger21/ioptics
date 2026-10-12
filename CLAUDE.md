# ioptics

基于 PyTorch 的集成光神经网络（Integrated Optical Neural Networks, IONN）仿真与训练框架。

## 环境管理 (Pixi)

本项目统一使用 **pixi** 进行依赖管理和任务执行。

- 安装并更新依赖：`pixi install`
- 运行测试：`pixi run test` （或 `pixi run pytest tests/`）
- 启动 Python 环境交互：`pixi run python`
- 激活开发环境 Shell：`pixi shell`
- 启动 JupyterLab：`pixi run jupyter lab`

## 仓库结构

- [ioptics/](ioptics/)：核心源码包
  - [components.py](ioptics/components.py)：光电子器件模型（PhaseShifter, Beamsplitter, MZI, Laser, Detector, TIA, ADC 等）
  - [layers.py](ioptics/layers.py)：光网格层（ClementsMesh, FCArray 等拓扑）
  - [models.py](ioptics/models.py)：模型顶层容器（OpticalNN）与训练流程
  - [differentiators.py](ioptics/differentiators.py)：可微分优化器与求导策略（ParameterShift, Adjoint, Bernoulli 等）
  - [backward_functions.py](ioptics/backward_functions.py)：底层反向传播逻辑
  - [simulator.py](ioptics/simulator.py)：硬件级非理想性仿真引擎（Simulator）
  - [nonlinear.py](ioptics/nonlinear.py)：光学非线性激活层
- [tests/](tests/)：单元测试与回归测试套件
- [examples/](examples/)：Jupyter 教程与示例
- [OPTIMIZATION_ROADMAP.md](OPTIMIZATION_ROADMAP.md)：优化路线图与核心改进方案

## 优化关键要点 (基于 OPTIMIZATION_ROADMAP.md)

1. **硬件仿真正确性与鲁棒性 (P0)**
   - 激光器模型中保留原始光场相位，增量叠加随机相位噪声；增加 `torch.clamp(min=0.0)` 防止负功率开方导致 NaN。
   - 探测器散粒噪声按通道独立计算方差（$\text{Var} = 2 q I \Delta f$），去除不符合物理规律的全局 `mean()`。
   - 重构 `OutputDomain` 自动推断机制，确保电场与功率转换安全，严禁复数流向 ADC 导致 `torch.clamp` 报错。
   - 规范光电链路物理量纲（输入光功率基准与 SI 制物理常数对齐），修复 ADC 异常默认高压量程（60V -> 2.0V），避免过大量化步长完全吞噬物理噪声和小信号死神经元。

2. **测试与 Benchmark 基准建立 (P0)**
   - 移除测试中的 `pytest.skip`，修复复数输入缺陷。
   - 引入硬数值断言：Adjoint 导数与 Autograd 严格对比，引入 `gradcheck` 梯度验证。
   - 补充 MZI 与 ClementsMesh 理想状态下的酉矩阵能量守恒测试（$U^\dagger U = I$）。
   - 建立不同规模（$N=4, 8, 16, 32, 64$）的前向/反向耗时基线。

3. **核心计算路径向量化 (P1)**
   - 消除 `MeshColumn` 与 `MZI` 中 Python 逐器件 `for` 循环与张量切片就地赋值，实现整列批处理构建。
   - 优化小 Batch 输入下的前向光场逐列传递，避免显式 $O(L \cdot N^3)$ 矩阵链乘。
   - 向量化 Adjoint 伴随法梯度求导。
   - 重构蒙特卡洛仿真（`run_mc`），利用 PyTorch Batch 维度实现多样本并行化。

4. **缓存与架构收敛 (P2)**
   - 引入静态转移矩阵缓存（`_cached_transfer_matrix`）。
   - 将 TIA / ADC 非理想性计算下沉至器件类，统一组件与仿真器职责。
