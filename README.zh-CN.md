**[English](README.md)** · 简体中文

<p align="center">
  <img src="docs/assets/jacobian-hero.jpg" width="100%" alt="一位数学家在黑板前工作的黑白档案风格照片，画面表现了雅可比行列式为常数以及三个不同输入映射到同一输出。">
</p>

<h1 align="center">Jacobian</h1>

<p align="center">
  <strong>为智能体提供原子化数学能力：发现一个类型化操作、运行它，并组合其有界结果。</strong>
</p>

<p align="center">
  <a href="https://github.com/morluto/jacobian/actions/workflows/ci.yml"><img src="https://github.com/morluto/jacobian/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/jacobian/"><img src="https://img.shields.io/pypi/v/jacobian" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/jacobian"><img src="https://img.shields.io/npm/v/jacobian" alt="npm"></a>
  <a href="https://pypi.org/project/jacobian/"><img src="https://img.shields.io/pypi/pyversions/jacobian" alt="支持的 Python 版本"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/morluto/jacobian" alt="MIT 许可证"></a>
</p>

Jacobian 是一个为 AI 智能体提供高等数学工具的 MCP 服务器，提供原子化、可组合的
数学计算。它只暴露两个工具：`math.find` 在不可变的类型化数学操作库中进行搜索，
`math.run` 精确执行其中一个操作并返回其具体的类型化结果。每个操作都是一次有界、
精确的计算：输入一个类型化请求，输出一个类型化数学值。同一套数学库也可通过 CLI
和原生 Python API 使用。

## 快速开始

无需全局安装 Jacobian，即可运行标准的 Python MCP 命令：

```sh
uvx --from jacobian jacobian-mcp
```

如果 MCP 主机需要 npm 命令，npm 包会作为确定性的载体调用同一个命令：

```sh
npx jacobian mcp
```

如需持久安装：

```sh
python -m pip install jacobian
jacobian-mcp
```

该软件包包含 Jacobian 所维护的完整 Python 后端栈：SymPy、NetworkX、Z3 和
Python-FLINT。因此，普通的 Python 或 npm 安装都会提供同一套内置的 Python
数学操作。经过测试的二进制安装契约是 glibc Linux x86-64 上的 CPython 3.12
或 3.13；发布门禁会在这两个 Python 版本上安装构建好的 wheel 并启动 Jacobian。
其他系统可能有兼容的上游 wheel，但目前不属于经过测试的发布契约。特别是，
Alpine/musl 无法从 PyPI 安装完整的必需后端栈。

Python 发行包包含数学内核、CLI 和 MCP 服务器。npm 包会确定性地将其精确版本映射
到相应的 `uvx` 调用。

## 计算一个有界结果

普通操作优先直接返回数学结果。例如，`matrix.determinant.compute` 接受一个
精确有理数矩阵，并直接返回其行列式。调用方可以将类型化结果传递给后续操作来
组合计算。

## 可用的数学能力

内置操作覆盖以下领域：

- 多项式映射和多项式代数；
- 精确线性代数；
- 图、路径、着色和同构；
- 有界 SAT 和 SMT 求解；
- 有限代数、概率、几何和拓扑；以及
- Lean 源代码 elaboration。

SAT 和 SMT 操作直接使用维护良好的 Z3 Python 绑定。可选的 `lean.check` 操作会在
固定的 Lean 服务环境中运行一段有界源代码，使用请求范围内的临时目录并返回类型化
诊断信息。使用 `math.find` 搜索操作、浏览陌生领域，并在调用 `math.run` 前检查单个
操作。

请参阅[领域操作库](docs/reference/domain-operation-library.md)了解维护中的操作
组合，并参阅[后端要求](docs/how-to/backend-requirements.md)。

## 状态

Jacobian 0.12.0 仍处于预稳定阶段。已发布的软件包和操作契约描述了受支持的
接口；实验性操作契约可能在不同版本之间变化。

## 文档

- [文档首页](docs/index.md)：教程、操作指南、参考资料和说明
- [架构](docs/explanation/architecture.md)：运行时结构和信任边界
- [产品模型](docs/explanation/product-blueprint.md)：操作契约、所有权和项目边界
- [工具参考](docs/reference/tools.md)：MCP 资源和调用契约
- [后端要求](docs/how-to/backend-requirements.md)：维护中的 Python
  后端和可选的 Lean
- [远程部署](docs/how-to/deploy-remote-mcp.md)：HTTP 部署和身份验证

## 贡献

Jacobian 使用 Python 3.12、`uv` 和一个精简的 `Makefile`：

```sh
make setup
make test-math
make check
```

修改代码前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。其中介绍了聚焦测试命令、
验证规则、文档归置和拉取请求要求。

## 许可证

[MIT](LICENSE)
