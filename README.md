# 化工 AI 灰箱建模：从机理方程到可训练模型

本书结合当前仓库 [HybridML](../README.md) 的模型描述和示例，用 Notebook 学习如何把化工机理与数据驱动模型组合起来。所有新增示例使用本地生成的数据，不需要下载数据集。

第 7–8 章的工程实现放在仓库根目录的 [`industrial_greybox/`](../industrial_greybox/README.md) 包中，可直接用 `python -m industrial_greybox.cli --quick` 跑通端到端流程，或用 `python -m unittest discover -s industrial_greybox/tests -t .` 运行测试。

## 学习目标

- 区分机理模型、纯数据模型和灰箱模型。
- 用物料衡算建立一个 CSTR 的稳态模型。
- 用现代 Keras API 实现可微分的机理层与数据驱动修正项。
- 阅读 HybridML 的模型 JSON，理解神经网络、ODE 和节点连线。
- 认识训练集内插、工况外推和实验验证的差别。
- 把教学级代码升级为可审查的工程实现：数值稳定性、化验死时间、工业数据清洗、慢变参数与影子运行。

## 推荐学习顺序

1. 阅读[灰箱建模思路](chapters/01-greybox.md)，先熟悉模型假设和参数含义。
2. 运行[仓库模型导读 Notebook](chapters/02-repository.ipynb)，对照现有示例理解 HybridML 的 JSON 结构。
3. 运行[CSTR 灰箱建模 Notebook](chapters/03-cstr.ipynb)，从合成数据开始训练并比较三类模型。
4. 按[下一步练习](chapters/04-next-steps.md)替换假设、检查外推并规划真实实验验证。
5. 运行[非等温动态 CSTR Notebook](chapters/05-nonisothermal-cstr.ipynb)，在完整运行轨迹上比较机理、灰箱与黑箱模型。
6. 阅读[HybridML 源码到工厂建模](chapters/06-hybridml-industrial-greybox-walkthrough.ipynb)，理解框架结构如何映射到工艺问题。
7. 阅读[准生产级工业灰箱系统](chapters/07-industrial-greybox.md)，了解数值稳定性、时滞补偿、数据清洗、慢变参数与影子运行的工程实现（代码见 [`industrial_greybox/`](../industrial_greybox/README.md)）。
8. 运行[准生产级灰箱系统实操](chapters/08-industrial-workflow.ipynb)，用合成 campaign 走通清洗、对齐、训练、诊断与影子运行。

## 环境与运行

优先使用仓库已有的 `.conda` 环境。当前工作区没有该环境时，在仓库根目录创建一次 Python 3.12 环境（TensorFlow 官方支持 Python 3.9–3.12），之后始终复用它：

```bash
conda create --prefix ./.conda python=3.12 pip -y  # 仅首次需要
conda run --prefix ./.conda python -m pip install -r tutorials/requirements.txt
```

如果 `.conda` 已存在，跳过 `conda create`，只执行安装命令。安装前可用 `conda run --prefix ./.conda python --version` 检查环境。安装后打开 Notebook：

```bash
conda run --prefix ./.conda python -m jupyter lab
```

要构建网站，进入 `tutorials/` 后运行：

```bash
cd tutorials
conda run --prefix ../.conda myst build --html
```

本书用 MyST（Jupyter Book 当前采用的工具链）构建。首次构建需要联网下载网站主题。若只想学习代码，可以直接在 JupyterLab 打开 `.ipynb` 文件。

## 关于仓库示例的依赖

仓库本身锁定了 TensorFlow 2.3.1 和 TensorFlow Probability 0.11.1 等旧依赖。它们不一定能安装在当前 Python 或 TensorFlow 环境中；仓库中的 `casadi_linear_ode` 也需要相应的 CasADi 依赖。因此本书把这些模型作为代码库的建模案例来讲解，而新增 CSTR Notebook 使用独立、现代的 Keras 3 API。

请不要把教程依赖直接装进旧版 HybridML 环境中。若需要运行原始 HybridML 模型，应依据该项目的依赖文件创建单独的兼容环境。
