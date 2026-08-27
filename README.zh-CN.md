**[English](README.md)** · 简体中文

<p align="center">
  <img src="docs/assets/jacobian-hero.jpg" width="100%" alt="黑白档案照：黑板上雅可比行列式为常数，三个不同输入映到同一输出。">
</p>

<h1 align="center">Jacobian</h1>

<p align="center">
  <strong>为智能体提供可执行的数学词汇：发现一个类型化操作，运行它，并组合其结果。</strong>
</p>

<p align="center">
  <a href="https://github.com/morluto/jacobian/actions/workflows/ci.yml"><img src="https://github.com/morluto/jacobian/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/jacobian/"><img src="https://img.shields.io/pypi/v/jacobian" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/jacobian"><img src="https://img.shields.io/npm/v/jacobian" alt="npm"></a>
  <a href="https://pypi.org/project/jacobian/"><img src="https://img.shields.io/pypi/pyversions/jacobian" alt="支持的 Python 版本"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/morluto/jacobian" alt="MIT 许可证"></a>
</p>

Jacobian 是一个 MCP 服务器，为 AI 智能体提供一套可搜索、类型化的数学操作。`math.find` 用来发现操作，`math.run` 每次只执行一个有界契约，并返回类型化结果。同样的数学库也可以通过 CLI 和原生 Python API 直接使用。

每个操作只确立一个稳定、可复用的数学后置条件，而不是规定工作流或证明策略。声称精确的地方就会保持精确，近似、不完备或不确定性也会显式标出。

**Jacobian 的假设是：数学推理会受益于一套语义范围明确、有界且可执行的词汇。** 与其提供庞大的领域求解器或预先排好的工作流，Jacobian 只提供可搜索、可组合的数学原语——智能体可以把它们拼成超出单个操作设计目标的解法。库负责提供可信的数学步骤，模型决定怎么选、怎么组合、何时停。操作保持语义范围明确并由所属领域维护，避免把某种证明策略固化到工具里。

更多关于语义原子性的讨论，见[可执行的数学词汇](docs/explanation/executable-mathematical-vocabulary.md)。

## 快速开始

用一条命令为你的智能体配好 Jacobian，需要 Node.js 20.17+、22.13+ 或 23.5+，并确保 `uvx` 在 `PATH` 中：

```sh
npx jacobian@latest setup
```

它会检测已安装的智能体，让你在写入前确认改动。它不会帮你安装 Node.js、Python、`uv` 或智能体本身。需要自动化预览时，可用 `npx jacobian@latest setup --codex --dry-run` 查看计划；只有在明确指定 `--codex`、`--all` 等参数时才加 `--yes`。

不用全局安装，也可直接运行标准的 Python MCP 命令：

```sh
uvx --from jacobian jacobian-mcp
```

若 MCP 客户端只能通过 npm 命令启动，npm 包会原样转发到同一条命令：

```sh
npx jacobian mcp
```

需要长期安装时：

```sh
python -m pip install jacobian
jacobian-mcp
```

这个包自带了 Jacobian 维护的完整 Python 后端栈：SymPy、NetworkX、Z3 和 Python-FLINT。所以无论是常规的 Python 安装还是 npm 安装，你拿到的都是同一套基于 Python 的内置操作。官方测试过的环境是 glibc Linux x86-64 上的 CPython 3.12 / 3.13，发版时会在这两个版本上安装构建好的 wheel 并启动验证。其他系统也可能有可用的上游 wheel，但不在官方测试范围内。特别地，Alpine/musl 无法从 PyPI 装全必需的后端。

Python 发行包里包含了数学内核、CLI 和 MCP 服务器。npm 包本身不提供 JavaScript API，只是把当前 npm 包的精确版本映射到对应的 `uvx` 调用。

## 计算一个有界结果

普通操作直接返回数学结果。例如，`matrix.determinant.compute` 接收一个精确的有理数矩阵，直接返回它的行列式。你可以把类型化的结果传给下一个操作，串起来完成更复杂的计算。

在本地终端中，先查看已安装操作的精确契约，再运行其中一个示例：

```sh
jacobian inspect integer.compute.extended_gcd
jacobian run integer.compute.extended_gcd --json '{"left":"84","right":"30"}'
```

第二条命令会以 JSON 返回最大公约数和 Bézout 系数。在 MCP 主机中，用
`math.find` 查看同一份契约，再用相同形状的 payload 调用 `math.run`。具体的
智能体调用流程见[发现和调用操作](docs/how-to/invoke-domain-operations.md)。

## 可用的数学能力

内置能力覆盖：

- 多项式映射与多项式代数；
- 精确线性代数；
- 图、路径、着色与同构；
- 有界 SAT 和 SMT 求解；
- 有限代数、概率、几何与拓扑。

SAT 和 SMT 操作直接调用 Z3 的 Python 绑定。用 `math.find` 搜索操作、浏览不熟悉的领域，先查看单个操作的契约，再用 `math.run` 执行一次。

请参阅[领域操作库](docs/reference/domain-operation-library.md)了解操作契约与准入规则，用 `math.find` 浏览实际的操作目录，并参阅[后端要求](docs/how-to/backend-requirements.md)。

## 状态

Jacobian 0.15.0 <!-- x-release-please-version --> 仍为预稳定版本。已发布包和操作契约即为受支持的接口，实验性契约可能在后续版本中调整。

## 文档

- [文档首页](docs/index.md)：教程、操作指南、参考和原理说明
- [架构](docs/explanation/architecture.md)：运行时结构与信任边界
- [产品模型](docs/explanation/product-blueprint.md)：操作契约、归属和项目边界
- [工具参考](docs/reference/tools.md)：MCP 资源与调用契约
- [后端要求](docs/how-to/backend-requirements.md)：维护的 Python 后端
- [远程部署](docs/how-to/deploy-remote-mcp.md)：HTTP 部署与身份认证

## 贡献

Jacobian 使用 Python 3.12、`uv` 和精简的 `Makefile`：

```sh
make setup
make handoff LANE=math TESTS=tests/math/graphs/test_graph_distance_matrix.py
```

改代码前请先读 [CONTRIBUTING.md](CONTRIBUTING.md)，里面有聚焦的测试命令、校验规则、文档组织和 PR 要求。

## 许可证

[MIT](LICENSE)
