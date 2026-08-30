# 明天计划（HackSprint London）

周日 30 日 · EF HQ Shoreditch · 10:30 开场 · **12:00 关门** · **17:00 提交**  
必须用 Daytona 才有奖。环境已通：venv、SDK 0.207、API key、eu。

---

## 叙事（忘了先看这里）

**我们是什么**  
GhostData = ML 数据管线的对抗式持续集成（CI = Continuous Integration，每次 PR 自动跑测试、不过就合不了）。

公司已经有：标注回放数据、冻结下游模型、现有 pipeline、他们自己信任的那套检查。  
GhostData 问的不是「数据违不违规」，而是：

> 能不能找到一个**说得通的管线故障**，现有检查全过，但下游模型坏了？

找到了，那就是一只 **Ghost**。

**Pitch（背下来）**

- 主句：Your tests can only catch failures you encoded. GhostData searches for the next one.
- 备选：Move the postmortem before the incident.
- 画面中心：Same values. Different relationships.（值都合法，意义坏了。）

**一只 Ghost 要交付成什么**

不是一句吓人的故事，是四样工程产物：

1. 可复现的变换（代码）  
2. 一份会让模型变差的数据集  
3. 模型影响报告（真 AUC，不许编）  
4. 一条新的回归测试 / data contract（下次 PR 合不进去）

**演示要讲的因果**

```text
Pipeline PR  →  现有检查 27/27 PASS
        ↓
   GhostData 搜索（每条假设 = 一台 Daytona 沙箱）
        ↓
   检查仍 PASS  +  模型 AUC 掉   =  Ghost
        ↓
   Promote → 将来的 PR 被挡住
```

英雄失败默认用 **实体/特征错位**：同一段人里把 income 对调。直方图、均值、缺失率都不变，但 income↔债务↔违约 的关系断了。这就是「现有检查看不见、模型依赖的东西坏了」。

**我们不是什么（评委一问就答）**

| 他们会说 | 你说 |
|---|---|
| 这不就是 drift 监控？ | 监控看线上真实数据；我们是部署前主动搜能骗过检查的反例。 |
| 这不就是 Great Expectations？ | 他们提供检查；我们把那些检查当约束，去找第 28 条。 |
| 这不就是乱加噪声 / 对抗样本？ | 不。只在已知生产故障库里搜（错位、过期特征、哨兵、重复）。 |
| 为什么要 Daytona？ | 每条候选可执行、隔离、用完即扔、可独立计量。 |

**禁止说**

- 没人做数据校验 / 没人做压力测试  
- GhostData 能预测未来故障  
- 每只 Ghost 都一定现实  
- Google/Uber 发生过我们编的那个 ACME 事故  

要说：我们在**显式约束下搜索说得通的反例**。

---

## 只做 P0

1. 公开信贷/欺诈表 + 冻结二分类器 + 可复现 baseline AUC  
2. 检查：schema / 空值 / 范围 / 均值 / PSI / 类别占比  
3. 四种失败：实体错位、特征过期、条件哨兵、重复  
4. Daytona runner：候选 → 隔离沙箱 → 真检查 + 冻结模型 → JSON  
5. 搜索：铺开 → 拒绝 → 排序 → 收细 → 赢家  
6. 三个界面：Pipeline PR / Search / Ghost Found  
7. 赢家升级成回归测试（挡住下次 PR）

P0 通了再加：LLM 选失败、自动解释、更漂亮的图。  
不做：监控平台、鉴权、时序、视觉、LLM eval。

---

## 时间

| 时段 | 干什么 |
|---|---|
| 到场–12:30 | 烤 snapshot + Volume；确认能并行开 4 台沙箱 |
| 12:30–14:30 | runner + 四种算子 + 真 AUC（数字不许编） |
| 14:30–16:00 | 搜索环 + 三个界面，Search 屏要能看见 Daytona worlds |
| 16:00–16:45 | 90 秒演示跑通；Preview 报告页如果来得及 |
| 16:45–17:00 | 提交 |

---

## Daytona 怎么用（详见 `DAYTONA_SHORT_zh.md`）

必用：并行 ephemeral 沙箱、声明式 Snapshot、Volume、解释器、文件上下传、图表产物、出站封锁。  
加分：Preview、label 列出正在跑的世界。  
别碰：GPU、桌面/VNC、Warm Pool。

---

## 演示 90 秒

1. 27/27 PASS → 点 RED-TEAM  
2. 4–6 个 Daytona world 在跑  
3. Ghost：检查仍过，AUC 真的掉  
4. 「同样的值，不同的关系」  
5. Promote to regression test → 下次 PR 被挡住

睡吧。到场先开一台沙箱，别一来就写前端。
