# Implementation Plan

- [x] 1. 先验实验E0：环境→链路解释力验证
  - [x] 1.1 加载Intel Lab trace数据并预处理
    - 解析data/Intel_Lab_Data/中的原始数据
    - 提取humidity、temperature、RSSI、LQI等字段
    - 按时间窗口分段，确保统计独立性
    - _Requirements: 1.1, 1.2_
  - [x] 1.2 计算环境特征与链路质量的相关性
    - 计算Pearson r和Spearman ρ
    - 计算滞后相关性（cross-correlation）
    - 报告p值和置信区间
    - _Requirements: 1.1, 1.2_
  - [x] 1.3 训练链路成功/失败预测器
    - 使用logistic regression预测链路成功
    - 报告AUC和Brier score
    - 进行置换检验验证显著性
    - _Requirements: 1.3, 1.4_
  - [x] 1.4 生成E0证据图
    - 绘制环境变化→可靠性下降的散点图/热力图
    - 符合MDPI规范（宽度>=1200px）
    - _Requirements: 1.5, 11.1_

- [x] 2. 先验实验E1：CAS特征贡献度验证
  - [x] 2.1 定义oracle mode计算逻辑
    - 实现U = PDR - λ·Energy效用函数
    - 扫描λ参数（0.1, 0.5, 1.0, 2.0）
    - 为每轮计算oracle最优模式
    - _Requirements: 2.1_
  - [x] 2.2 拟合可解释模型
    - 使用logistic regression从特征预测oracle mode
    - 提取系数符号和显著性
    - 计算permutation importance
    - _Requirements: 2.2, 2.3_
  - [x] 2.3 生成E1特征重要性图
    - 绘制特征系数条形图
    - 标注显著性水平
    - _Requirements: 2.4, 11.1_

- [x] 3. 先验实验E2：Safety阈值概率论标定
  - [x] 3.1 实现Beta-Binomial模型
    - 将每轮delivery建模为Bernoulli试验
    - 拟合Beta先验
    - 计算后验P(p < θ | data)
    - _Requirements: 3.1, 3.2_
  - [x] 3.2 优化阈值以控制误触发率
    - 扫描θ和T参数组合
    - 计算每组合的false positive rate
    - 选择FPR < 10%的最优组合
    - _Requirements: 3.3, 3.5_
  - [x] 3.3 生成E2阈值标定图
    - 绘制FPR vs θ/T曲面图
    - 标注选定的工作点
    - _Requirements: 3.4, 11.1_

- [x] 4. 先验实验E3：负载均衡验证
  - [x] 4.1 计算负载分布指标
    - 实现Gini系数计算
    - 实现Jain's fairness index计算
    - 从round_statistics提取gateway/CH负载
    - _Requirements: 4.1_
  - [x] 4.2 分析负载与性能关系
    - 计算负载均衡度与PDR/能耗的相关性
    - 计算效应量和置信区间
    - _Requirements: 4.2, 4.3_
  - [x] 4.3 生成E3负载分析图
    - 绘制负载分布直方图
    - 绘制负载-性能散点图
    - _Requirements: 4.4, 11.1_

- [x] 5. 先验实验E4：MCU决策时延验证
  - [x] 5.1 加载并分析benchmark_decision_time.json
    - 提取各规模的决策时延数据
    - 计算统计量（mean, std, percentiles）
    - _Requirements: 5.1_
  - [x] 5.2 生成ECDF和scaling曲线
    - 绘制决策时延ECDF图
    - 绘制时延随规模增长曲线
    - 标注MCU预算线（25ms）
    - _Requirements: 5.1, 5.2_
  - [x] 5.3 与ML/RL方法对比
    - 收集文献中ML/RL方法的时延数据
    - 生成对比表格
    - _Requirements: 5.3, 5.4_

- [-] 6. 扩展实验矩阵执行
  - [x] 6.1 实现实验矩阵执行器
    - 定义场景×规模×负载的完整矩阵
    - 实现并行执行逻辑（支持8-16 workers）
    - 实现结果聚合和存储
    - _Requirements: 6.1, 6.2, 6.3_
  - [ ] 6.2 运行Intel replay实验
    - 使用真实Intel Lab trace
    - 运行30+ seeds
    - 记录完整指标体系
    - _Requirements: 6.4_
  - [ ] 6.3 运行合成室内实验
    - uniform/corridor/cluster/obstacle拓扑
    - 100/300/500节点规模
    - 低/中/高/bursty负载
    - _Requirements: 6.1, 6.2, 6.3_
  - [ ] 6.4 运行动态压力实验
    - moving BS/dropout/phase shift场景
    - 记录失败模式诊断信息
    - _Requirements: 8.2, 8.5_

- [-] 7. 大规模网络PDR诊断与修复
  - [ ] 7.1 添加详细诊断日志
    - 记录跳数分布
    - 记录网关负载
    - 记录骨干使用率
    - _Requirements: 7.1_
  - [x] 7.2 参数敏感性分析
    - 扫描k_gw（2,4,6,8）和k_sk（2,3,4,5）
    - 记录每组合的PDR
    - 生成敏感性曲面图
    - _Requirements: 7.4_
  - [x] 7.3 修复或明确适用范围
    - 如果300节点PDR可修复至>50%，更新参数
    - 否则在论文中明确声明"支持≤100节点"
    - _Requirements: 7.2, 7.3_

- [x] 8. 统计验证管道
  - [x] 8.1 实现Welch t检验
    - 对所有协议对进行检验
    - 记录t值、p值、自由度
    - _Requirements: 10.1_
  - [x] 8.2 实现效应量计算
    - 实现Cliff's δ（非参数）
    - 实现Hedges g（参数）
    - 为所有对比计算效应量
    - _Requirements: 10.1_
  - [x] 8.3 实现Bootstrap置信区间
    - 使用BCa方法
    - 10000次重采样
    - 计算95% CI
    - _Requirements: 10.1_
  - [x] 8.4 实现Holm-Bonferroni校正
    - 对所有p值进行校正
    - 生成校正后的显著性表
    - _Requirements: 10.2_

- [x] 9. CAS模块效应量评估
  - [x] 9.1 运行消融实验
    - 分别禁用CAS/Gateway/Safety/Fairness
    - 记录各配置的PDR和能耗
    - _Requirements: 9.2_
  - [x] 9.2 计算各模块效应量
    - 使用Hedges g计算效应量
    - 按贡献度排序
    - _Requirements: 9.1, 9.2_
  - [x] 9.3 重新定位CAS模块
    - 如果CAS效应量< 0.3，调整论文定位
    - 将Gateway/Safety作为核心创新
    - _Requirements: 9.3_

- [x] 10. MDPI规范图表生成
  - [x] 10.1 统一图表风格系统
    - 设置全局matplotlib参数
    - 确保宽度>=1200px
    - 设置svg.fonttype='none'
    - _Requirements: 11.1, 11.2_
  - [x] 10.2 生成主文关键图（6-8张）
    - 先验实验证据图（E0-E4）
    - 主实验对比图
    - 消融实验图
    - 敏感性分析图
    - _Requirements: 11.1_
  - [x] 10.3 运行图表验证
    - 使用validate_figures.py检查
    - 修复不符合规范的图表
    - _Requirements: 11.4, 11.5_

- [x] 11. 代码质量清理



  - [ ] 11.1 移除冗余代码
    - 识别未使用的函数和类
    - 清理废弃的导入

    - _Requirements: 12.1_
  - [ ] 11.2 重构核心协议文件
    - 将aeris_protocol.py拆分为模块
    - 确保单文件行数< 1000
    - _Requirements: 12.2_

  - [x] 11.3 运行静态检查




    - 使用ruff/flake8检查
    - 修复E/W级别错误
    - _Requirements: 12.3_


- [x] 12. 论文更新与提交准备





  - [x] 12.1 更新论文定位


    - 根据实验结果调整声称
    - 删除不支持的动态场景声称



    - 强调轻量级/MCU可部署性
    - _Requirements: 8.1, 8.3_
  - [x] 12.2 完善补充材料


    - 敏感性分析完整表格
    - 全量统计检验矩阵
    - 更多拓扑实验结果



    - _Requirements: 10.3_
  - [x] 12.3 生成最终提交包
    - LaTeX源文件
    - 所有图表（PDF/SVG/PNG）
    - 补充材料PDF
    - 复现脚本和数据
    - _Requirements: 10.3, 12.4_
