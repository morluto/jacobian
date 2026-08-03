<p align="center">
  <img src="docs/assets/jacobian-hero.jpg" width="100%" alt="一位数学家在黑板前工作的档案风格黑白照片；图中展示了一个行列式恒定的雅可比映射，以及映射到同一输出的三个不同输入。">
</p>

<h1 align="center">Jacobian</h1>

<p align="center">
  <strong>为智能体提供可执行的数学能力，让独立检查器可以重放证据。</strong>
</p>

<p align="center">
  面向猜想、反例、精确计算和形式化证明的 MCP 服务器、CLI 与 Python 库。
</p>

<p align="center">
  <a href="https://github.com/morluto/jacobian/actions/workflows/ci.yml"><img src="https://github.com/morluto/jacobian/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/jacobian/"><img src="https://img.shields.io/pypi/v/jacobian" alt="PyPI"></a>
  <a href="https://www.npmjs.com/package/jacobian"><img src="https://img.shields.io/npm/v/jacobian" alt="npm"></a>
  <a href="https://pypi.org/project/jacobian/"><img src="https://img.shields.io/pypi/pyversions/jacobian" alt="支持的 Python 版本"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/morluto/jacobian" alt="MIT 许可证"></a>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <a href="#quickstart">快速开始</a> ·
  <a href="#verification">验证</a> ·
  <a href="#capabilities">能力</a> ·
  <a href="#documentation">文档</a> ·
  <a href="#contributing">贡献</a>
</p>

Jacobian 为 AI 智能体提供小型、可组合的数学操作，而不是一个不透明的通用求解器。智能体可以构造对象、计算不变量、搜索见证，并将精确证据提交给独立的检查器。每一步都会以类型化结果或 artifact 的形式保持可见。

信任边界是有意设计的：搜索结果、求解器状态、模型答案、超时或评分都不会直接升级为 `VERIFIED`。只有经过操作者授权的检查器才能生成已验证记录，并且该记录必须绑定到确切的命题、候选对象、范围、语义、证书格式和检查器身份。

<a id="quickstart"></a>

## 快速开始

npm 启动器会安装 Jacobian，并配置受支持的 MCP 客户端。若只想临时使用而不进行全局安装，可以运行：

```sh
npx jacobian setup
```

如果需要反复使用，可以持久安装启动器并使用其命令：

```sh
npm install -g jacobian
jacobian setup
jacobian upgrade
jacobian doctor
```

Python 发行版可以直接安装稳定版本：

```sh
python -m pip install jacobian
```

启动器支持 Claude、Codex、Cursor、Gemini 和 OpenCode。它需要 Node.js 18 或更高版本、Python 3.12 以及 [`uv`](https://docs.astral.sh/uv/)。运行 `jacobian mcp` 可以直接启动服务器。

<details>
<summary><strong>从源代码安装</strong></summary>

```sh
git clone https://github.com/morluto/jacobian.git
cd jacobian
./scripts/setup-agent --client codex --profile full-python --yes
```

该命令会执行锁定的完整 Python 环境同步，并配置所选智能体，让 MCP 从绝对源代码路径和状态路径启动，同时使用 `--no-sync`。它还会记录一份 doctor 报告，其中包含 Git 修订版本、包版本、catalog 摘要和 provider 可用性。关于 `core`、`full-python`、`lean` 和 `external-proof` 配置档案、试运行、可重复性及回滚行为，请参阅[从源代码检出配置智能体](docs/how-to/setup-agent-from-source.md)。

使用 `uv run jacobian --help` 查看 CLI，或使用 `uv run jacobian-mcp` 启动 MCP 适配器。

</details>

<a id="verification"></a>

## 验证如何工作

Jacobian 将寻找证据与判断证据能证明什么分开。假设智能体正在检验命题 **“`F` 是单射。”**

<p align="center">
  <img src="docs/assets/verification-flow.jpg" width="100%" alt="单射命题引出一个候选碰撞，随后进行精确的独立检查并生成验证记录；缺少见证、超时、取消和错误都仍然表示未知。">
</p>

**命题 → 候选见证 → 独立检查 → 验证记录**

| 阶段 | 输出 | 它建立了什么 |
| --- | --- | --- |
| 命题 | `F` 是单射 | 需要调查的陈述；尚未可信 |
| 搜索 | 候选见证 `(F, p, q)` | 可检查的证据，而不是结论 |
| 独立检查 | 精确确认 `p ≠ q` 且 `F(p) − F(q) = 0` | 候选对象确实构成碰撞 |
| 记录 | 将已检查的碰撞绑定到原始命题和检查器身份 | 单射命题为 `FALSE · VERIFIED` |

> **没有见证不等于证明。** 搜索失败、超时、取消或错误都会使命题保持为 `UNKNOWN`。

在入门教程中，同一个边界表现为：

```text
evaluate.batch   →  FALSE  · HEURISTIC
witness.find     →  精确的见证 artifact
witness.verify   →  FALSE  · VERIFIED
```

`FALSE · HEURISTIC` 是一次评估；`FALSE · VERIFIED` 是由独立检查过的证据支持的结论。请参阅[查找并验证反例](docs/tutorials/first-verified-result.md)中的可运行示例。

<a id="capabilities"></a>

## 能力

能力通过 `capability://catalog` 在运行时发现，使用 `capability.describe` 描述，并使用 `capability.invoke` 执行。已安装的 catalog 是事实来源，因为可用性可能取决于本地 backend。

| 领域 | 面向智能体的结果 |
| --- | --- |
| 多项式映射 | 求值映射、计算雅可比矩阵、搜索碰撞、独立验证碰撞 |
| 多项式代数 | 规范化类型化表达式、分解一元多项式、验证恒等式、验证精确的方程组解 |
| 精确线性代数 | 计算行列式、秩、核和整数行 Hermite 标准形；为 `Ax = b` 查找并独立验证有理数解或不相容证书 |
| 图 | 构造和检查图、枚举路径、实现度序列、测试同构、搜索着色 |
| SAT 与 SMT | 查找模型或证明 artifact；独立重放赋值、DRAT 证明和 Alethe 证明 |
| 泛代数 | 求有限幺半群律的值并搜索反模型 |
| 多胞体 | 计算凸组合和线性分离 |
| Lean | 发现声明、获取前提、检查证明状态，并在固定环境中检查证明 |
| 研究记忆 | 存储带有修订版本的草稿、发现、尝试、关注点和依赖关系上下文 |

请参阅[工具参考](docs/reference/tools.md)了解公共接口，参阅[原子能力组合](docs/contributing/atomic-capability-portfolio.md)了解能力组合设计和评估门槛。

## 设计

Jacobian 将四项职责分开：

- **策略由智能体负责。** 内核提供数学操作，不规定研究工作流。
- **能力暴露一个连贯的结果。** 有用的中间对象、失败信息和证明义务保持可见。
- **值可以直接组合。** 小型、有界的数学结果保持内联；artifact 用于可复用对象、可重放证据和大型载荷。
- **检查器负责信任。** 插件和搜索代码不能授权检查器，也不能更改验证策略。

公共 MCP 接口保持精简：能力 catalog，以及两个能力入口 `capability.describe` 和 `capability.invoke`。

<a id="documentation"></a>

## 文档

| 从这里开始 | 需要详细信息时 |
| --- | --- |
| [文档主页](docs/index.md) | 教程、操作指南、参考和设计说明 |
| [架构](docs/explanation/architecture.md) | 系统结构和独立验证边界 |
| [产品模型](docs/explanation/product-blueprint.md) | 能力契约、所有权、artifact 和 assurance |
| [产品目标](docs/explanation/goals.md) | 当前优先事项和研究方向 |
| [工具接口](docs/reference/tools.md) | MCP 资源、工具和调用契约 |
| [领域操作库](docs/reference/domain-operation-library.md) | 内置 producer、有界搜索、artifact 和精确重放契约 |
| [Provider 运行时](docs/reference/provider-runtime.md) | backend 可用性、兼容性和身份 |
| [测试策略](docs/reference/testing-strategy.md) | 验证层、命令和 CI 职责 |

此外还有专门的 [SAT artifact](docs/reference/sat-artifacts.md)、[SMT/Alethe artifact](docs/reference/smt-artifacts.md)、[精确有理数线性系统证据](docs/reference/linear-rational-solutions.md)、[精确有理数矩阵行列式](docs/reference/matrix-rational-determinant.md)、[整数矩阵 HNF](docs/reference/matrix-hermite-normal-form.md) 和 [Lean 声明发现](docs/reference/lean-declaration-discovery.md)契约。[领域能力操作指南](docs/how-to/invoke-domain-capabilities.md)演示发现、计算调用、有界结果解释和精确重放；[Lean 形式化中间结果参考](docs/reference/lean-formal-intermediates.md)介绍证明状态、前提获取、依赖图和已检查编辑。

## MCP 客户端与部署

`jacobian setup` 会向一个或多个受支持的客户端注册本地服务器。`jacobian upgrade` 会刷新启动器所管理环境中的固定 Python 内核；若要升级 npm 启动器本身，请使用 `npm install -g jacobian@latest`。

对于代码仓库副本，`jacobian setup --source <checkout> --state-dir <path> --profile full-python` 会显式地将客户端绑定到该源代码环境；受维护的 `scripts/setup-agent` 封装器会先完成所需的锁定同步和 doctor 检查。

服务器只公布能力入口；`capability.describe(query=...)` 会先搜索已安装的精简结果，智能体再检查精确契约并调用它。这是一个工具箱接口：数学分解、探索和组合由智能体负责。

支持 MCP resource 的客户端可以读取 `jacobian://instructions` 获取操作指南，读取 `capability://catalog` 获取完整的机器可读 inventory。支持 prompt 的客户端还可以选择请求 `jacobian-discover` 或 `jacobian-check-evidence`，以获得协议脚手架。

远程客户端可以通过带 bearer token 认证和 subject 绑定租户状态的 Streamable HTTP 或 SSE 连接。请参阅[部署远程 MCP 服务器](docs/how-to/deploy-remote-mcp.md)。静态 token 适用于受控部署，不是托管身份系统。

在 systemd 主机上的干净 clone 中，受维护的安装器可以部署 localhost endpoint、由 Caddy 管理的公共域名或 Tailscale Funnel：

```sh
sudo ./deploy/install.sh
sudo ./deploy/install.sh --mode domain --domain math.example.org
sudo ./deploy/install.sh --mode tailscale
```

运行 `./deploy/install.sh --help`，或添加 `--dry-run` 先查看计划。公共模式需要经过检查的 Caddy 安装；Funnel 还需要已连接的 Tailscale 安装。认证默认启用，新生成的 bearer token 只会打印一次。

## 可选 backend

部分能力使用默认未安装的 backend：

- CaDiCaL 用于查找 SAT 模型和 UNSAT 证明 artifact。
- cvc5 生成 SMT UNSAT 证明；Carcara 独立检查 Alethe。
- `flint` extra 提供 Python-FLINT/Arb 操作，用于精确有理数系统、整数矩阵与格、多项式以及经过验证的数值计算。具体能力和独立重放支持取决于已安装的 catalog。
- 固定版本的 Lean `CORE` 和 `MATHLIB` 环境用于检查形式化证书。

Backend 可用不等于验证授权。Provider 的输出在相应的独立检查器接受其绑定的见证或证书之前，始终保持未验证状态。

<details>
<summary><strong>Lean 证书</strong></summary>

`lean.check` 能力会将精确命题和证明正文绑定到结果。随附环境固定了 Lean、imports 和允许的 trust base；模型提供的 imports 和 package 会被拒绝。

使用以下命令准备固定版本的运行时：

```sh
elan toolchain install leanprover/lean4:v4.31.0
cd lean
lake update
lake build
```

证明状态交互和前提获取是探索辅助工具。只有成功的 `lean.check` 才能让其输出成为 `VERIFIED`。请参阅[声明发现引导教程](docs/tutorials/lean-declaration-discovery.md)。

</details>

<details>
<summary><strong>macOS 与 Z3</strong></summary>

锁定环境使用 `z3-solver` 5.0.0.0。其上游 macOS wheel 面向 macOS 13 或更高版本，支持 Apple silicon 和 Intel。在更旧的系统上，`uv` 会退回源码构建，这需要 CMake、`make` 和 C++20 编译器。

重试 `uv sync --dev` 前，请安装 Xcode Command Line Tools 和 CMake。以下命令只报告相关环境，不会修改环境：

```sh
sw_vers -productVersion
uname -m
xcode-select -p
clang++ --version
cmake --version
make --version
```

关于上游 wheel 标签，请参阅 [PyPI 上的 `z3-solver` 5.0.0.0 文件](https://pypi.org/project/z3-solver/5.0.0.0/#files)。

</details>

## 状态

Jacobian 0.6.0 是预稳定版本。已发布的 package、能力和 artifact 契约描述当前受支持的接口；持续进行的能力研究可能会在不同版本之间改变实验性契约。

Python 发行版包含数学内核、CLI 和 MCP 服务器。npm package 是用于启动该实现并安装 MCP 客户端的轻量启动器，不是独立的 JavaScript API。

<details>
<summary><strong>关于主视觉图</strong></summary>

这幅图的视觉主题来自雅可比猜想的三维反例：一个精确恒定的雅可比行列式，以及三个映射到同一输出的不同有理数输入。这些方程很好地概括了 Jacobian 的目的——令人意外的候选对象很有价值，但只有精确计算和独立检查才能确定哪些内容值得信任。

Terence Tao 撰写了[一篇易读的数学介绍](https://terrytao.wordpress.com/2026/07/21/a-digestion-of-the-jacobian-conjecture-counterexample/)。行列式恒等式和碰撞也已经在 Isabelle/HOL 中[独立形式化](https://isa-afp.org/entries/Jacobian_Counterexample.html)。二维猜想仍然是开放问题。

</details>

<details>
<summary><strong>项目边界</strong></summary>

Jacobian 的目标不是将通用数学本体、自然语言到形式数学的翻译器、分布式搜索基础设施或不透明的通用求解器放入内核。它不会重新实现定理证明器或 SAT/MIP 求解器，不接受任意由模型提供的可执行 bundle，也不会把浮点评分、超时和求解器标签当作证明。

</details>

<a id="contributing"></a>

## 贡献

Jacobian 使用 Python 3.12、`uv` 和一个小型 `Makefile`：

```sh
make setup
make test-unit
make check
```

修改代码前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)。其中说明了聚焦测试命令、验证规则、文档放置方式和 pull request 要求。

## 许可证

[MIT](LICENSE)
