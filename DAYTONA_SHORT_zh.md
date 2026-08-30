# Daytona 短记

Daytona 现在卖的不是 IDE，是给 Agent 用的**可编程隔离电脑（Sandbox）**。你的程序负责思考；它负责真的跑命令、改文件、起服务、看桌面。

入口：[文档](https://www.daytona.io/docs/en/) · SDK `pip install daytona`（当前 0.207）

---

## 主流能力（产品主叙事）

1. **Sandbox**  
   一台远程隔离机：独立内核、文件系统、网络、CPU/内存/磁盘。默认 Linux 容器，毫秒级创建。另外有 Linux VM、Windows、GPU。

2. **跑代码**  
   shell、`code_run`（Python/TS/JS）、有状态 Python 解释器、后台进程、PTY。这是最核心的一层。

3. **文件系统**  
   读写、上传下载、搜索替换。Agent 的工作区。

4. **Snapshot**  
   拍环境模板（依赖、文件）。之后从快照开新沙箱，等于克隆工位。可用 SDK 声明式构建镜像，不必自己推 registry。

5. **Volume**  
   多沙箱共享磁盘：数据集、模型、缓存。沙箱删了数据还在。

6. **Git / LSP**  
   clone、commit、push；语言服务（补全/诊断）。偏「在沙箱里改仓库」。

7. **Preview / 网络**  
   内部端口变成预览 URL；出站可防火墙 / 白名单。给人看 Agent 刚起的 Web App，同时限制乱访问外网。

---

## 较新、官方在推的大功能

- **Computer Use / VNC**：键盘、鼠标、截图，让 Agent 操作 GUI（Linux / Windows / macOS 桌面）。
- **GPU**：NVIDIA（H100 / H200 / RTX 等），偏推理和训练；官方定位是临时沙箱。
- **Secrets**：密钥不进沙箱明文，出口代理按白名单主机替换。
- **Warm Pool**：按快照预热一批已在跑的沙箱，创建时直接领走（秒开）。文档有，本账号 list 接口曾 404，演示别赌这个。
- **VM 专属**：Pause/Resume、Fork、带内存的热快照。容器没有这些。
- **MCP**：官方 MCP，Cursor / Claude 等可直接当工具后端用，不必手写 SDK。
- **声明式 Image**：SDK 里 `pip_install` 构建快照，官方强调「Everything Docker provides, now in Daytona」。

---

## 怎么记

| 以前 | 现在 |
|---|---|
| 给人开 Codespaces | 给 Agent 开隔离电脑 |
| IDE / SSH 进去写一天 | SDK / API / MCP，用完即删或打快照 |
| 一个项目一个长期环境 | 任务级并行，评测 / Agent / 解释器 |

对已有 Agent：Daytona 是第 3 层（执行后端），不是再养一个对话模型。

---

## 明天 GhostData 要用哪些

评委要求 **必须用 Daytona**。不要只 `create` + `echo`。一个候选 = 一台沙箱，并行 4–6 台，用完即删。尽量叠新功能，但别为了用而用 GPU / 桌面。

**必用（评委能看见）**

- 并行 ephemeral Sandbox：每个 World 一台，带 label（world_id / 失败类型）
- **声明式 Image + Snapshot**：先烤好 `ghostdata-runner`（pandas / sklearn），演示时从快照开，不要当场 pip
- **Volume**：参考数据和冻结模型挂一次，各沙箱共享
- **跑代码**：`code_run` + 有状态解释器（先过检查，再打冻结模型）
- **文件系统**：上传 WorldSpec / 变换；下载 report.json、Ghost parquet
- **图表产物**：matplotlib 直方图 / 关系图（「同样的值，不同的关系」）
- **出站封锁** `network_block_all`：候选变换不可信，证明隔离不只是口号

**值得加（新功能、加分）**

- Preview：把 Ghost 报告起成静态页，给评委一个 URL
- Labels + list：搜索屏上「6 台正在跑」从 Daytona 列出来，不要假数据
- Secrets：如果沙箱要调任何外部 API，密钥走 Secrets，不要塞进 env

**不要用（和产品打架）**

- Computer Use / VNC / Windows / GPU
- 在沙箱里再开一个 coding agent（MCP 只给你自己调试，不是产品）
- Warm Pool：文档有，这台账号接口 404，演示别赌
- VM Fork/Pause：容器 + Snapshot 就够讲「克隆工位」

一句话对评委：GhostData 在搜反例；每一条假设都在独立 Daytona 沙箱里用真实检查和冻结模型跑完再销毁。
