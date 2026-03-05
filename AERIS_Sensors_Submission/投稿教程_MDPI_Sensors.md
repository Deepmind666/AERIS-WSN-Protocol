# MDPI Sensors 投稿教程（完整版）

---

## 一、投稿包清单

打开 `AERIS_Sensors_Submission/` 文件夹，确认以下文件全部存在：

```
AERIS_Sensors_Submission/
│
├── manuscript.tex              ← LaTeX 源文件
├── manuscript.pdf              ← 编译好的 PDF（22 页）
├── bibliography.bib            ← 参考文献（36 条，DOI 已验证）
│
├── cover_letter.txt            ← 正式投稿信（带信头、日期、收件人）
├── highlights.txt              ← 5 条论文亮点
├── abstract_plaintext.txt      ← 纯文本摘要（供粘贴到网站表单）
├── author_statements.txt       ← 作者贡献/基金/伦理/数据/利益声明
│
├── Definitions/                ← MDPI LaTeX 模板
│   ├── mdpi.cls
│   ├── mdpi.bst
│   └── ...（logo 等辅助文件）
│
└── figures/                    ← 10 张 PDF 矢量图
    ├── fig1.pdf    → Figure 1: AERIS 工作流程图
    ├── fig2.pdf    → Figure 2: 100 节点 PDR 对比
    ├── fig3.pdf    → Figure 3: 消融热力图 + 边际效应
    ├── fig4.pdf    → Figure 4: 可扩展性趋势
    ├── fig5.pdf    → Figure 5: 显著性热力图
    ├── fig6.pdf    → Figure 6: 功率敏感性 delta 图
    ├── fig7.pdf    → Figure 7: 绝对 PDR 剖面
    ├── fig8.pdf    → Figure 8: 压力-delta 证据
    ├── fig9.pdf    → Figure 9: 权衡面板
    └── fig10.pdf   → Figure 10: NS-3 趋势验证
```

> 图文件名 fig1~fig10 与论文正文 Figure 1~10 编号一一对应。

---

## 二、注册 MDPI 账号（首次投稿才需要）

1. 打开 **https://susy.mdpi.com/**
2. 页面 **右上角** → 点 **"Register"**
3. 填写：

| 字段 | 填写 |
|------|------|
| Email | `1403073295@mails.gdut.edu.cn` |
| First Name | `Kangrui` |
| Last Name | `Li` |
| Affiliation | `Guangdong University of Technology` |
| Password | 自设（≥8位，含大小写+数字） |

4. 勾选同意条款 → 点 **"Register"**
5. 去邮箱点验证链接 → 激活成功

---

## 三、进入投稿系统

1. 打开 **https://susy.mdpi.com/** → 点 **"Login"** → 输入账号密码
2. 进入后看到 Dashboard 仪表盘页面
3. 点页面中央的 **绿色大按钮 "Submit a Manuscript"**
   - 位置：页面正中偏上，非常醒目
   - 或者点左侧菜单栏 **"Submit"**

---

## 四、选择期刊和类型（第 1-2 页）

### 页面 1：选期刊
1. **Journal** 下拉框 → 输入 `Sensors` → 选 **"Sensors (ISSN 1424-8220)"**
2. 点右下角 **"Next"**

### 页面 2：选文章类型
1. **Article Type** → 选 **"Article"**
2. **Section** → 选 **"Sensor Networks"**
   - 下拉框在页面中部，可能需要滚动
   - 找不到 "Sensor Networks" 也可选 "Internet of Things"
3. 点 **"Next"**

---

## 五、填写稿件元数据（第 3-7 页）

### 页面 3：Title & Abstract

| 字段 | 操作 |
|------|------|
| **Title** | 粘贴：`AERIS: Environment-Aware Hierarchical Routing for Reliable Wireless Sensor Networks under Realistic Channel Conditions` |
| **Abstract** | 打开 `abstract_plaintext.txt`，复制 "Abstract:" 后面的全部文本粘贴 |
| **Keywords** | 逐个输入，每输一个点 **"Add"** 按钮 |

Keywords 依次输入（共 6 个）：
```
wireless sensor networks
reliable routing
hierarchical protocol
packet delivery ratio
scalability
reproducible evaluation
```

点 **"Next"**

### 页面 4：Authors

点 **"Add Author"** 按钮，添加 3 位作者。

**第 1 位（通讯作者）：**

| 字段 | 填写 |
|------|------|
| First Name | `Kangrui` |
| Last Name | `Li` |
| Email | `1403073295@mails.gdut.edu.cn` |
| Affiliation | `Faculty of Automation, Guangdong University of Technology, Guangzhou 510006, China` |
| ORCID | 如有填写，没有留空 |
| **Corresponding Author** | **✅ 勾选**（这个勾很重要！） |

**第 2 位：**

| 字段 | 填写 |
|------|------|
| First Name | `Xiaobo` |
| Last Name | `Zhang` |
| Email | 张老师邮箱 |
| Affiliation | 同上 |

**第 3 位：**

| 字段 | 填写 |
|------|------|
| First Name | `Junyi` |
| Last Name | `Lin` |
| Email | 林同学邮箱 |
| Affiliation | 同上 |

点 **"Next"**

### 页面 5：Manuscript Details

| 字段 | 操作 |
|------|------|
| **Cover Letter** | 打开 `cover_letter.txt`，全文粘贴到文本框 |
| **Number of Figures** | 填 `10` |
| **Number of Tables** | 填 `15` |
| **Supplementary Materials** | 选 `No` |

点 **"Next"**

### 页面 6：Funding, Ethics, Conflicts

| 字段 | 操作 |
|------|------|
| **Funding** | 填 `This research received no external funding.`<br>（如有基金写：`This work was supported by xxx [grant number xxx].`） |
| **IRB Statement** | 选 `Not applicable` |
| **Informed Consent** | 选 `Not applicable` |
| **Data Availability** | 填 `Data available on request from the corresponding author.` |
| **Conflicts of Interest** | 填 `The authors declare no conflict of interest.` |

> 以上内容也写好在 `author_statements.txt` 中，直接复制粘贴。

点 **"Next"**

### 页面 7：Suggested Reviewers

MDPI 要求填 **3~5 位** 建议审稿人：

| 要求 | 说明 |
|------|------|
| 不能是同单位 | 不能填广工的老师 |
| 不能是合作者 | 近 3 年没合作过 |
| 需要提供 | 姓名、邮箱、单位 |
| 建议选择 | WSN routing / IoT protocol 方向的教授 |

> 如果暂时不确定人选，部分期刊允许先跳过。

点 **"Next"**

---

## 六、上传文件（最关键的一页）

### 上传方式

MDPI 系统有一个文件上传区域，页面上有 **"Browse..."** 或 **"Drag & Drop"** 区域。

### 必须上传的文件

| 顺序 | 选择文件 | File Type 下拉选 | 说明 |
|------|---------|-----------------|------|
| 1 | `manuscript.pdf` | **Manuscript** | 主稿 PDF |
| 2 | `fig1.pdf` | **Figure** | Description 填 `Figure 1` |
| 3 | `fig2.pdf` | **Figure** | Description 填 `Figure 2` |
| ... | ... | **Figure** | 依次到 fig10.pdf |
| 12 | `fig10.pdf` | **Figure** | Description 填 `Figure 10` |

> **注意**：初次投稿只上传 PDF。LaTeX 源码在被接受后 Production 阶段才需要。届时把整个 `AERIS_Sensors_Submission/` 文件夹打成 ZIP 上传。

每上传一个文件后，确认列表中显示 **绿色勾号 ✅**。

全部上传完 → 点 **"Next"**

---

## 七、预览和提交

### 步骤 1：预览
- 系统自动合并生成预览 PDF
- 点 **"Preview"** 按钮
- 逐页检查：标题、作者、摘要、图片、表格、参考文献

### 步骤 2：确认声明
页面底部有 **3 个复选框**，全部勾选：

- [x] 本稿未在其他地方发表或投稿
- [x] 所有作者已阅读并同意提交
- [x] 同意 MDPI 开放获取政策

### 步骤 3：提交
- 点页面 **右下角** 的 **"Submit"** 按钮（蓝色/绿色大按钮）
- 弹出确认对话框 → 点 **"Confirm"**

### 步骤 4：完成
- 页面显示 **"Submission Successful"**
- 通讯作者邮箱收到确认邮件
- 记下 **Manuscript ID**，格式如 `sensors-3456789`

---

## 八、投稿后时间线

| 阶段 | 预计时间 | 你需要做什么 |
|------|---------|-------------|
| Editorial Check | 1-3 天 | 等待，不用做任何事 |
| Peer Review | 2-4 周 | 等待，可以在系统查看状态 |
| Decision 邮件 | 审稿后 3-5 天 | 看邮件和系统通知 |

### 常见 Decision 类型

| 结果 | 含义 | 你要做什么 |
|------|------|-----------|
| **Accept** | 直接接受 | 等 Production 阶段校对 |
| **Minor Revision** | 小修 | 7-14 天内按意见改完回传 |
| **Major Revision** | 大修 | 需要较大改动，可能再审 |
| **Reject** | 拒稿 | 考虑改投其他期刊 |

### 查看投稿状态
1. 登录 **https://susy.mdpi.com/**
2. 左侧菜单 → **"Submissions"**
3. 点你的稿件 Manuscript ID 查看当前阶段

---

## 九、Revision（修稿）怎么做

如果收到 Minor/Major Revision：

1. 下载审稿意见 PDF
2. 写 **Response Letter**（逐条回复每个审稿人的每个意见）
3. 修改 manuscript，用红色标记改动部分
4. 登录系统 → 找到稿件 → 点 **"Submit Revision"**
5. 上传修改后的 PDF + Response Letter

---

## 十、费用说明

| 项目 | 金额 |
|------|------|
| 投稿费 | 免费 |
| APC（文章处理费） | 2600 CHF ≈ ¥20,000 |
| 收费时间 | 文章被 Accept 后 |
| 减免 | 可咨询学院是否有机构折扣 |

> APC 只在接受后才收取，投稿和审稿阶段不花钱。
