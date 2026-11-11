# 实施与验证记录

| Rung | 功能 | 验证命令 | Tier | Status |
|---|---|---|---|---|
| F0 | 真实视频入口到元数据产物 | `python -m person_tracking inspect --out person_tracking/runs/inspection.json` | MUST | 通过 |
| F1 | 预训练人员检测与逐路跟踪 | `python -m person_tracking track --start-clock 15:57:20 --duration 30 --out person_tracking/runs/smoke` | MUST | 通过执行检查 |
| F2 | 四路分组采样与候选标签 | `python -m person_tracking sample --interval 30 --out person_tracking/data/person_v1` | MUST | 通过执行检查 |
| F3 | 人工复核及受控数据集导出 | `python -m pytest person_tracking/tests -q` | MUST | 通过，14 项 |
| F4 | 文档、运行脚本及结果核对 | `python -m person_tracking doctor` | MUST | 通过 |

## Run record

- 2026-11-05：无已有代码、无 Git 仓库；不初始化 Git，不修改原视频/设计说明。
- F0：使用现有 camera-review 环境运行，退出码 0。四路均为 1920×1080、25 fps，时长 4120.36–4138.48 秒；产物 `runs/inspection.json`。无占位模型或模拟结果。
- F1：独立 crane-person 环境已安装，CUDA 12.4/PyTorch 2.5.1 在 RTX 3060 Laptop GPU 可用。四路各 300 分析帧完成检测、跟踪和回放，退出码 0；4 项时序/状态回归通过。小车俯视画面确认轨迹少，执行通过不代表准确率达标。已把每批重复的 half 弃用提示修正为此版本支持的 quantize=16/32。
- 依赖修复：defaults 索引缺少 pip，改克隆现有 camera-review；只在新环境通过 conda 卸载 OpenCV，再安装指定 pip 版本，避免无 RECORD 文件冲突。CUDA 大包改分段下载，未改原环境。
- F2（历史版本）：四路按 30 秒间隔采样，排除时间组边界，完成 442 张原尺寸图片和候选框，进程退出码 0。随机 seek 有 H.264 引用帧警告。原检查仅核对 OpenCV 返回的位置号，不足以证明与从头累计的帧号相同；需要复核图像质量及时间元数据。未将候选框或空候选自动转成人工真值。
- F3：14 项回归通过，包括原帧号和时间、安装映射、短遮挡续接、逐路隔离、低吞吐按秒超时、同步时段不跨集合、空标签必须确认、未复核不导出、分组泄漏拒绝、导出路径及副本隔离、HTTP 取图和写标签。真实数据未写任何复核通过标记，测试只使用临时合成样本。
- F3 页面：Node 语法检查退出码 0；本机浏览器实开页面、查看候选框、切换样本，均显示正常，真实数据仍为 442 待复核。测试服务器已停止。
- F4：doctor 退出码 0，CUDA/GPU 可用；pip check 无依赖破损；冻结环境记录在 `environment-freeze.txt`。PowerShell 入口四路 5 秒无视频复跑退出码 0，quantize 参数无弃用刷屏。
- H.264 修复：D01 直接 seek 可复现 12 行参考帧警告；实际累计帧号/25 与 PTS 有差异，68500 帧的 PTS 为 2746.134 秒。默认改为顺序解码并按 PTS 选帧，显式区分 seek 估计帧号和实际累计帧号。20 项回归通过；D01 原报警时段 1 秒/10 帧实际检测、前 2 秒/4 张实际采样均退出码 0、stderr 0 行。同为 PTS 2740 秒的顺序/seek 图像完全一致，仅证明该目标帧未被改变。原样本和历史回放未覆盖，未训练。详见 `H264解码警告排查与修复_2026-11-05.md` 及 `runs/h264_diagnosis/`。

## 审查状态

`SWEEP_UNAVAILABLE`：当前工具未接入技能要求的独立跨模型审查接口。未伪造审查结论；本次只报告实际运行、测试和人工可查看的回放产物，不报告模型正确率或生产验收通过。见 `SILENT_ASSUMPTION_SWEEP.json`。

## 占位实现

无。尚未进行的训练、边缘盒部署、跨相机身份和三维业务均在范围之外，未通过占位结果替代。

## Deferred

- 模型训练：此前明确暂不训练，且当前没有人工真值。
- RK3588 实机部署及端到端性能验收：本次环境是 Windows 工作站，未连接边缘盒。
- 跨相机身份识别、真实三维位置与 B1–B10 事件：本次请求是人员检测跟踪基础代码。

## Blockers

暂无。
