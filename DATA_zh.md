# 下了哪些数据、各自干什么

全部是**公开的信贷/审批表**，不是股价、不是银行生产库。  
GhostData 需要「一行一个人 + 好/坏标签」，才能训冻结模型、再故意搞坏数据。

三个文件夹 = 三种用法：

- **build**：明天写代码、看分布、调 Ghost，随便打开  
- **blind**：字段可以知道，**先别看数字/别拿来调参**。App 能跑通后再测「没见过的行」  
- **live**：最后换一张完全不同的表，测适配器死不死  

---

## 1. Give Me Some Credit（Kaggle 2011）— 主菜

**文件**

- `data/build/givemesomecredit.csv` — 12 万行，明天做 demo 用这张  
- `data/build/givemesomecredit_debug_3k.csv` — 从上面抽的 3000 行，跑通 pipeline 用，省时间  
- `data/blind/gmsc_holdout.csv` — 同一张表另外 3 万行，**同一套列**，用来盲测「换一批人行不行」

**每一行是什么**  
一个消费信贷客户（美国那边的零售信用，竞赛数据）。

**要预测的标签**  
`SeriousDlqin2yrs`：未来两年会不会严重逾期（大约 90 天以上没还）。

**列是干什么的**

| 列 | 人话 |
|---|---|
| MonthlyIncome | 月收入。英雄案例：对调一部分人的收入 |
| DebtRatio | 负债相对收入。和收入一起构成「关系」 |
| RevolvingUtilizationOfUnsecuredLines | 信用卡等循环额度用了多少（0～很大） |
| age | 年龄 |
| NumberOfTime…PastDue… | 以前逾期过几次（30/60/90 天档） |
| NumberOfOpenCreditLinesAndLoans | 现在有几条贷款/信用卡 |
| NumberRealEstateLoansOrLines | 房贷/不动产贷款条数 |
| NumberOfDependents | 家属人数 |

**为什么选它**  
公开表里少见「收入 + 负债 + 违约」三件套，正好讲：直方图不变，模型依赖的关系坏了。

**局限**  
2011、没有「自雇」字段、收入缺大约两成。演示用，不是 2026 实盘。

---

## 2. German Credit（德国 Statlog）— 小表调试

**文件** `data/build/german_credit.csv`（1000 行）

**每一行**  
一笔德国银行贷款申请。

**标签** `class`：good / bad（这份贷款好不好）。

**和 GMSC 的差别**  
有 `employment`、`job`、`purpose`、`credit_amount`。更像「工种/用途」叙事，但只有 1000 行，AUC 会抖。  
**用途**：本地秒级跑通检查和变换，不要当评委主表。

---

## 3. UCI 台湾信用卡违约（2005）— 另一套字段的盲测

**文件** `data/blind/uci_credit_default.csv`（3 万行）

**每一行**  
一个台湾信用卡客户。论文常用公开集。

**标签** `default.payment.next.month`：下个月会不会违约。

**列是干什么的（和 GMSC 完全不同）**

| 列 | 人话 |
|---|---|
| LIMIT_BAL | 信用额度 |
| SEX / EDUCATION / MARRIAGE / AGE | 人口统计 |
| PAY_0 … PAY_6 | 近几个月还款状态（拖了几个月） |
| BILL_AMT1 … | 账单金额 |
| PAY_AMT1 … | 实际还款金额 |

**用途**  
App 好了之后：同一套 GhostData，换「没有 MonthlyIncome」的表，看检查/模型适配器会不会写死列名。  
**不是**收入对调那个英雄故事（这张表没有月收入）。

---

## 4. Credit Approval / Australian Credit — 最后换皮压测

**文件**

- `data/live/credit_approval.csv`（690 行）  
- `data/live/australian_credit.csv`（690 行）

**是什么**  
老的「批不批信用卡」公开集。列被故意改成 `A1, A2, …`，不告诉你哪一列是收入。

**标签**  
`credit_approval` 有 `class`（批 / 拒）。Australian 的最后一列 `A15` 一般是同类标签。

**用途**  
只测一件事：代码有没有写死 `MonthlyIncome`。能读任意表、指定「哪列是标签、哪列拿来对调」就算过。  
**不要**用来讲评委故事，列名没法讲「收入」。

---

## 明天你实际碰哪几份

1. 写代码、90 秒演示：只碰 **Give Me Some Credit 的 build 两份**。  
2. 想快速迭代：先 `debug_3k`，再上 12 万行。  
3. German 仅当「还有工种」的备用小实验。  
4. blind / live：提交前或 App 稳了再开，当没见过的表。
