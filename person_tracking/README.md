# 现场人员检测、持续跟踪与训练数据准备

部署位置：`E:\横向项目-目标检测\person_tracking`。所有示例在项目根目录 `E:\横向项目-目标检测` 的 PowerShell 中执行。默认 Python：`D:\Anaconda\envs\crane-person\python.exe`。截至 2026-11-06，反例修正后的 v4 YOLO11n 已完成 80 轮，训练期最佳第 64 轮；独立 val 复评见下文。测试集仍封存，模型尚未部署到边缘盒。

**本轮反例复核与 YOLO11n 重训已完成。**800 张全部已复核，固定 train/val/test=560/117/123，导出到 `data/person_reviewed_v4_20261106/`，测试集保持封存。完整 80 轮训练产物位于 `runs/train/person_yolo11n_static_v4_20261106_full80/`；第 64 轮最佳权重的独立 val mAP50/mAP50–95 为 62.52%/26.83%，固定阈值 0.10 的 P/R 为 78.12%/62.50%。详细对照、限制和参数见 [本轮重训记录](YOLO11n_反例修正重训记录_2026-11-06.md)。模型尚未切换跟踪入口，也未部署到边缘盒。标注页面 <http://127.0.0.1:8768> 的数据与训练导出相互独立；若继续修改标注，需要再次审计和导出新版本。下文 8766、8767 及旧版数量属于历史记录。

实际部署、训练结果和边缘盒限制见 [部署与验证记录](部署与验证记录_2026-11-05.md)。现有 442 张图片已全部人工复核：138 张有人、304 张确认无人，共 185 个人员框；train=261、val=102、test=79。已导出 `data/person_reviewed_v1/data.yaml`。预训练基线及首轮/80 轮续训模型的同口径验证见 [标注检查与数据集导出记录](标注检查与数据集导出_2026-11-05.md)；test 未用于训练、早停、选权或评估。`runs/smoke/` 保存首次四路各 30 秒的带框跟踪回放。

2026-11-06 已建立新标注版本 `data/person_v2_800_20261106/`：800 张中继承 442 张人工标注，新增 358 张待复核；train/val/test=547/174/79。当前二轮标注地址为 <http://127.0.0.1:8766>，默认仅显示待复核新增图。原 79 张 test 保留并锁定标签，原目录未改动；未开始新一轮训练。数量、标注规则、导出条件见 [数据集扩充与二轮标注说明](数据集扩充与二轮标注_2026-11-06.md)。

## 1. 直接运行

```powershell
Set-Location 'E:\横向项目-目标检测'

# 检查依赖、GPU、权重和配置
.\person_tracking\run.ps1 doctor

# 四路同一文件名时钟附近，运行 30 秒，输出位置必须是新目录
.\person_tracking\run.ps1 track --start-clock 15:57:20 --duration 30 --out person_tracking/runs/my_demo

# 在不改相机配置的情况下验证指定检查点
.\person_tracking\run.ps1 track --config person_tracking/configs/cameras_yolo11n_v4.json --cameras D01 --start-clock 15:57:20 --duration 30 --weights person_tracking/runs/train/person_yolo11n_static_v5_d01_hard_20261106/weights/best.pt --out person_tracking/runs/d01_v5_diagnostic

# 单路持续处理直到录像结束；Ctrl+C 可正常关闭并落盘
.\person_tracking\run.ps1 track --cameras D01 --out person_tracking/runs/d01_full

# 四路从头处理；只保存轨迹，不生成较大的回放视频
.\person_tracking\run.ps1 track --no-video --out person_tracking/runs/all_full

# 从已完成的 D01 全片推理中构造人工复核候选；全画面逐张复核，不训练
.\person_tracking\run.ps1 review-hard-negatives --base person_tracking/data/person_v4_negative_review_20261106 --inference person_tracking/runs/inference/d01_yolo11n_v4_20261106 --out person_tracking/data/person_v5_d01_hard_review_20261106

# 打开新增候选复核页，确认真实人员框，确认无人帧；吊钩、钢丝绳和背景都不标人员
.\person_tracking\run.ps1 review --dataset person_tracking/data/person_v5_d01_hard_review_20261106 --port 8769

# CPU 模式（速度会降低）
.\person_tracking\run.ps1 track --cameras D01 --start-clock 15:57:20 --duration 10 --device cpu --imgsz 640 --out person_tracking/runs/cpu_demo
```

也可直接调用 `& 'D:\Anaconda\envs\crane-person\python.exe' -m person_tracking ...`。`--start` 表示距各自文件开头的秒数；`--start-clock` 表示近似录像时刻，二者优先采用后者。

`--fps 10` 是目标分析取帧率；不会改变原视频文件。本地录像现在默认从头顺序解码，按视频呈现时间戳（PTS）选择每个目标时刻及之后的首个可解码帧；时间戳存在空档时，输出帧数可能少于“时长 × 10”。不能用标称 25 fps 替代实际视频时间。`--imgsz` 是模型输入尺度参数，不是相机码流选择。当前录像仍是 1920×1080，程序不会把录像声称为正式第三码流。

从较晚时刻开始也需要先解码前面的画面。例如本机 D01 解码到约第 45 分钟需要约 50 秒；四路首次出结果会更慢，终端会显示预读进度。需要快速预览时可显式添加 `--decode-mode seek`，但该选项可能再次触发下文的 H.264 警告。

## 2. 相机和模型配置

配置：`configs/cameras.json`；跟踪器配置：`configs/botsort.yaml`。

| 通道 | 位置 | 安装组 |
|---|---|---|
| D01 | 右侧人员侦测 | 大车 |
| D02 | 副钩吊点 | 小车 |
| D03 | 左侧人员侦测 | 大车 |
| D04 | 主钩吊点 | 小车 |

- 检测：YOLO11n，只保留类别 `0: person`。默认 `imgsz=1280`；优先保证远处小目标的输入像素数，后续用人工真值决定是否换更大模型或 ROI 精检。普通预训练模型对俯视、遮挡、只露安全帽的人员可能漏检。
- 跟踪：BoT-SORT 的卡尔曼预测、两级框匹配和 `sparseOptFlow` 背景运动补偿；不加载外观 ReID 模型。四路共享检测模型，各有独立跟踪器。
- 跟踪前执行保守的人员嵌套框去重：小框至少 90% 面积落在大框内，且小框面积不超过大框的 65% 时，仅保留较大的框。被抑制的小框不进入跟踪器，因此不会申请重复 ID；只是部分重叠的人员框仍分别保留。此规则不能区分极端重叠、完全嵌套的两名真实人员，需在多人遮挡画面上复核阈值。
- 默认分析 10 fps，丢失缓冲 2.5 秒，按各路设定分析帧率转换为跟踪帧数，并按源时间再次检查超时，避免直播处理变慢时延长身份保留时间。低置信度框可帮助已有轨迹续接；新 ID 需要达到 `new_track_thresh`。
- 新轨迹需连续 3 次被观测后才显示 ID；确认后在跟踪器保留期间维持已确认状态，短时漏检不再要求重新累计 3 帧。遮挡期间最多显示 2 秒的虚线预测框并标记 `PRED`；预测框不计入真实检测统计或 MOT 真值文件。失轨超过 2.5 秒关闭轨迹并允许后续建立新 ID。未确认的过期候选状态会清理，避免长时间运行累积。
- 标识人员应使用 `(camera_id, session_id, track_id)`。不同相机的 ID 1 不表示同一人，重启/重连后不继承身份。短遮挡有机会续接，超过缓冲或多人交叉仍可能换号。
- `roi: []` 表示尚未配置业务 ROI，不伪造黄色/梯形检测区。可填归一化多边形 `[[x/W,y/H], ...]`；程序继续检测全图，只为框底中心标记 `inside_roi`，不会因为 ROI 外漏检影响跟踪。
- D01 当前明确按整幅图像检测；ROI 不参与筛选。吊钩、钢丝绳、工件或固定背景若被持续识别为人员，阈值和轨迹确认只能抑制一部分短暂误报，不能替代针对性负样本。应将误报所在全帧加入人工复核，只框真实人员；确认无人帧保存为空标签。机器候选框不是真值，不可直接训练。
- 视频起始时刻来自文件名，`time_offset_s` 可以在核对烧录时钟后修正；它不代表硬件曝光同步，也不用于三维融合。

## 3. 输出与对接

跨进程命令行/JSONL 接口和供 Python 程序逐帧消费的 `run_person_tracking()` SDK 见[人员检测程序对外接口说明](对外接口说明.md)。

B3 安全帽佩戴/颜色、B4 躯干工装颜色的数据标注、专项推理和训练路线见[B3/B4 专项方案](B3B4安全帽与工装颜色专项方案.md)。

每次运行必须使用新的 `--out` 路径，避免覆盖已有结果。

```text
runs/my_demo/
  run_config.json         实际模型、设备、参数、run_id
  summary.json            执行状态及四路统计
  D01/                    D02、D03、D04 同结构
    annotated.mp4         默认 1280 像素宽的带框轨迹回放
    frames.jsonl          每个分析帧一条，保留无人帧
    mot.txt               只导出观测到且已关联的框，MOT 10 列格式
    tracks.jsonl          轨迹起止、观测次数、最大观测空档
    summary.json          本路执行统计
```

`frames.jsonl` 的 `source_frame` 是 **0 起始帧号**：默认顺序模式按实际成功解码的帧累计，`source_frame_basis=decoded_frame_count`；快速跳转后为 OpenCV 的位置估计，标记 `opencv_position_estimate_after_seek`，不能当作从头累计帧号。`analysis_frame` 是 **1 起始分析帧号**。`mot.txt` 第一列使用后者，评估时必须用相同采样序列准备真值，不可直接与原录像 25 fps 的帧号混用。

本地文件 `video_time_s` 来自视频 PTS，`time_basis=file_pts`；PTS 缺失、重复或倒退的帧才以标称帧率递推，标记 `nominal_fps_fallback`，本路摘要记录 `timestamp_fallback_frames`。PTS 仍不等于曝光时间或严格同步时钟。OpenCV 读不到下一帧时不能可靠区分正常结束与码流故障，摘要记录 `reader_finish_reason=eof_or_decode_failure`，应结合 stderr 和原片检查。

每个人员观测包含 `xyxy`、`confidence`、`bbox_bottom_center`、`track_id`、`observed` 和 `inside_roi`。坐标都对应**原始输入图像尺寸**，不是回放视频或模型缩放后的坐标。`track_id=null` 表示已检测到人员候选、尚未建立确认轨迹；不会因为没有 ID 而从日志中删除候选。

每帧还记录去重前的 `raw_detection_count` 与 `detections_suppressed_nested`；帧内 `detections` 和摘要 `detections` 是去重后数量。摘要 `raw_detections` 和 `detections_suppressed_nested` 可用于监控规则的影响范围。规则参数也写入 `run_config.json`。

`cameras_yolo11n_v4.json` 使用连续 3 次命中确认轨迹，显示框单独做平滑；2026-11-07 增加显示置信度滞回：达到 `display_min_confidence=0.20` 后显示，同一轨迹当前帧仍有检测且置信度不低于 `display_keep_confidence=0.15` 时继续显示。低于保留阈值后隐藏，恢复到 0.20 才重新显示；该滞回仅作用于真实观测框，遮挡预测框由独立预测逻辑绘制。没有配置 `display_keep_confidence` 的旧配置仍沿用单阈值行为。该规则只影响绘制，不改检测、跟踪关联或 MOT 坐标，也不等于提升检测 P/R。对既有 v4 D01 的 1,200 帧记录离线重放时，连续观测同一轨迹的显示开关次数从 42 次降到 18 次，显示观测数从 1,311 增至 1,328；新增的 17 个弱检测未经过人工正确性判定。审计见 `runs/audits/d01_display_hysteresis_20261107/report.json`，该审计未生成新回放视频。

框底边中心是二维位置近似。它不是鞋底接地点，不可直接用于离地高度、人员在吊物下方或跨相机三维比较。`frames.jsonl` 会同时记录真实观测和短时预测框，并用 `observed`/`predicted` 区分；预测框仅用于回放连续性，不写入 `mot.txt`，也不能作为当前帧已检测到人员的证据。

`tracks_created`、`detections` 和 `tracked_observations` 是程序执行计数，不能作为真实人数、准确率或跟踪成功率。轨迹时间跨度包括短暂丢失，不能直接当业务停留时间。

`tracks.jsonl` 的 `first_video_time_s` 和 `observed_frames` 从轨迹确认后开始记录，不包括确认之前的候选命中。因此 `observed_frames=1` 不表示检测器总共只观察到 1 帧，也不能直接据此判断单帧误报。

## 4. 从四路视频准备训练数据

```powershell
# 原分辨率均匀采样、按全局时间组划分并生成候选框；不训练
.\person_tracking\run.ps1 sample --interval 30 --out person_tracking/data/person_v2

# 按场景补充更密集的困难样本，另存新目录
.\person_tracking\run.ps1 sample --cameras D01 D02 D03 D04 --start-clock 15:57:20 --duration 60 --interval 2 --out person_tracking/data/hard_1557

# 继续本轮 800 张数据的人工标注；原 442 张状态已继承
.\person_tracking\run.ps1 review --dataset person_tracking/data/person_v2_800_20261106 --port 8766

# 查看复核进度
.\person_tracking\run.ps1 dataset-status --dataset person_tracking/data/person_v2_800_20261106

# 所有新增图复核或排除后，再导出到新目录；检查 pending=0，见二轮标注说明
$exportOut = "person_tracking/data/person_reviewed_v2_" + (Get-Date -Format "yyyyMMdd_HHmmss")
.\person_tracking\run.ps1 export --dataset person_tracking/data/person_v2_800_20261106 --out $exportOut
```

每张样本保留 `camera_id`、源文件、源帧号及其来源、视频 PTS 秒数及时间来源、近似拍摄时刻、划分组和候选框数量。默认顺序解码，采样和跟踪共用同一读取逻辑；根据实际采到的 PTS 分组，再统一分配 train/val/test。图片保存原分辨率 JPEG，不裁掉烧录时间或自动缩小。JPEG 是有损重编码，原始录像仍保留。

已有 `data/person_v1/` 和 `runs/smoke/` 等历史结果使用旧的“帧号 ÷ 标称帧率”时间基准及随机跳转，没有自动覆盖。它们的时间/帧号不能直接与新版顺序模式混用；现有 442 张图片仍可人工复核，但用于跨路时间对齐或正式验证前应重新检查元数据，必要时生成新版本。

数据目录中的 `images/` 是待标注图像，`proposals/labels/` 是**机器候选**，`reviewed_labels/` 才是已确认标签；`reviews.json` 记录审核/排除状态。候选框为空不表示已确认无人。

复核页面支持查看/删除候选框、拖动画新框、保存当前标注、明确确认无人、排除无法可靠标注的帧，以及按通道和审核状态筛选。复核时要查漏检，不能只检查已有框是否正确。样本只有头盔或手臂且无法可靠确认人员时，宜排除，不把明显人员残片标成负样本；可见截断人员按画面内可见外接框标注，遮挡严重的标注口径需统一。

划分采用四路共享的十分钟时间组、固定随机种子，约 70/15/15 分给 train/val/test。每组边界各排除 60 秒，避免四路同一时刻和紧邻帧跨集合。文件名时钟存在误差，划分仍须复核；同一天同一次作业的数据只能作为初步开发集，正式验收需另一天/另一作业批次独立视频。

均匀采样仅是第一批样本。应再补充俯视人员、远处小人、蹲姿/弯腰、多人交叉、吊物遮挡、进出画、运动模糊、强反光、运输平台上的人员和确实无人但像人的设备。新增视频再采样时需重新统一划分，不能把两个独立导出的数据集直接混合以为无泄漏。

## 5. 训练与验证路线

2026-11-06 反例修正后的 v4 人员检测已按用户选择使用 YOLO11n，并完成总计 80 轮续训。数据为 train/val/test=560/117/123，test 未推理。训练期最佳为第 64 轮；同一修正 val 的独立复评中，最佳权重 mAP50/mAP50–95=62.52%/26.83%，固定置信度 0.10 下 Precision/Recall/F1=78.12%/62.50%/69.44%（TP/FP/FN=25/7/15）。第 80 轮末权重为 64.73%/24.99%，固定阈值 P/R=84.62%/55.00%；因此报告和后续验证优先使用最佳权重。验证集仅 40 个真值人员框，结果属于开发验证，不代表独立日期泛化、跟踪能力或边缘盒性能。尚未切换运行入口或部署。详见 [YOLO11n 反例修正重训与验证记录](YOLO11n_反例修正重训记录_2026-11-06.md)。

**2026-11-07 当前状态：v9 重标训练已完成，未部署。** 对 363 张原有人框图片重新复核、518 张确认无人图片沿用原记录；18 个标签文件发生变化（train 17、test 1，val 0）。审计后保留原时间组成员，train/val/test=655/115/111。YOLO11n 从公开权重重新初始化，最多 80 轮、patience 15，第 36 轮早停，最佳第 21 轮。v7/v8/v9 的 115 张 val 图片和标签及独立验证器设置完全一致。v9 最佳权重复评 mAP50/mAP50–95=70.62%/29.67%，固定 conf=0.10、IoU=0.50 下 Precision/Recall=84.62%/57.89%（22/4/16 TP/FP/FN）；Precision 上升但 Recall 仍低于 v7 的 68.42%，D02/D03 漏检较多，未达到 Recall 85% 目标。test 未用于训练或评估，运行入口仍未切换。[v9 重标训练与验证记录](YOLO11n人员重标v9训练记录_2026-11-07.md)。

v7 D01 回放进一步暴露轻度遮挡后的漏检：显示平滑、轨迹确认和置信度滞回只能稳定已有检测，不能提升检测器 Recall。已生成 `data/person_v8_d01_occlusion_review_20261107/` 遮挡专项复核集，新增 31 张候选只进入 train，test 继续封存；复核页为 `http://127.0.0.1:8772/?status=pending`。空检测候选不能直接标为无人，必须人工查看后再导出和重训。

修订后的审计入口是 `finalize-corrections`，会生成独立冻结副本和导出集，保留原时间组归属并核对继承空标签；`finalize-hard-negatives` 要求基线标签不变，不适用于本轮修订。新审计还绑定干净公开预训练权重的 SHA-256，训练入口会拒绝从错误标签训练的旧检查点续训。具体命令见上述修订说明。

2026-11-06 标注修正后的 v3 重训已完成总计 80 轮：train 560 / val 117 / test 123，公开 YOLO11n 重新初始化，初段第 33 轮早停后补足 80 轮。最佳仍为第 18 轮，训练期 mAP50/mAP50–95 为 54.641%/22.347%；独立 val 复评为 55.847%/22.360%，固定阈值 P/R 为 70.000%/46.667%。最佳权重与早停时参数相同，更多轮次未改善最佳结果，test 继续封存。v2 错标训练保留为历史。详见 [修正标注后的人员重训记录](修正标注后的人员重训记录_2026-11-06.md)。

2026-11-05 已完成首轮静态 person 检测训练及总计 80 轮续训。首轮在第 42 轮早停、最佳权重对应第 27 轮；从第 40 轮检查点恢复到第 80 轮后，最佳权重仍是第 27 轮，参数与首轮最佳权重逐项相同。独立 val 复评中，best 的 mAP50/mAP50–95 为 71.63%/30.36%，固定置信度 0.10、匹配 IoU 0.50 时 Precision/Recall 为 54.76%/69.70%；last（第 80 轮）为 61.82%/25.54% 和 80.00%/60.61%，表现出精确率提高但召回率下降。样本量小、相机间结果不均衡，完整对比和限制见部署记录。本机跟踪入口仍使用 `configs/cameras.json` 中的公开预训练权重，微调权重未被自动切换，也未上边缘盒。

检测器只使用 train 学习、val 早停和选权；人员静态框不含跨帧身份、工件框或业务事件标签，因此本轮没有训练跟踪器或人员与工件关系模型。测试集保留用于未来独立验证。

## 6. D01 吊钩/背景困难样本微调

2026-11-06 已完成 56 张全画面困难样本复核并导出 v5：train/val/test=616/117/123，新增 D01 样本全部只进入既有 train 时间组；原 800 张、全部 test 标签和图像保持锁定。审计和逐文件 SHA-256 清单位于 `runs/audits/person_v5_d01_hard_20261106_r3/`。以 v4 best 初始化、等待 RTX 3060 空闲后训练：

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m person_tracking.experiments.train_person_v2 --dataset person_tracking/data/person_reviewed_v5_d01_hard_20261106_r3 --out person_tracking/runs/train/person_yolo11n_static_v5_d01_hard_20261106 --weights person_tracking/runs/train/person_yolo11n_static_v4_20261106_full80/weights/best.pt --device 0 --epochs 80 --patience 15 --wait-for-idle
```

训练按每 5 轮记录当前 val mAP50/mAP50–95 和最佳轮次；本轮因 patience=15 在第 38/80 轮早停，best 为第 23 轮。训练期 best mAP50/mAP50–95=57.32%/25.19%，最终在同一 val 的独立复评为 58.23%/24.57%。

v4 best 与 v5 best 使用相同 117 张 val、40 个人员框复评，test 未参与：

| 模型 | mAP50 | mAP50–95 | 固定阈值 Precision | Recall | TP/FP/FN |
|---|---:|---:|---:|---:|---:|
| v4 best | 62.52% | 26.83% | 78.13% | 62.50% | 25/7/15 |
| v5 best | 58.23% | 24.57% | 57.14% | 60.00% | 24/18/16 |

固定工作点为置信度 0.10、匹配 IoU 0.50。v5 相比 v4 未改善，仍低于 Precision/Recall 均须超过 85% 的目标，不切换当前运行模型。v5 分相机 P/R：D01 50.00%/66.67%（3 个真值框）、D02 60.00%/60.00%（5 个）、D03 45.45%/45.45%（22 个）、D04 81.82%/90.00%（10 个）；D03 误检与漏检都最突出。样本量小且来自单日现场视频，属于开发验证，不代表独立日期泛化或现场安全验收。

v5 的 Precision/Recall 均未达到 85%，已触发按短板追加人工标注。后续版本仍用独立 val 上固定置信度 0.10、匹配 IoU 0.50 的 P/R 判断，不用训练集指标或训练过程自动选择的 PR 点判定；test 继续封存。当前数据集是单类 `person`，YOLO 标签不支持在人员框上附加“负框”属性：精确率不足时，从**不属于 val/test 的新视频时间段**抽取吊钩、钢丝绳等误检全帧，用户逐帧复核并只标真实人员；无人帧保存为空标签，画面中其他非人目标留空即可作为背景负样本。召回率不足时，应补标漏掉的真实人员框，重点覆盖远处小人、遮挡和人员与吊物交叠；仅加无人/背景负样本不能修复漏检。如果确实需要把吊钩等干扰物逐个画框标记，应作为额外正类别（例如 `hook`）一致标注，训练多类别模型并在运行时只输出 `person` 类，不能把干扰物框伪装成负标签。若把当前 val 中的样本并入训练，必须重新建立独立验证划分，不能再把该 val 当作独立评估结果。

针对 D01 吊钩/背景误检，先用 `review-hard-negatives` 从已有全片推理结果中按时间分层抽取候选帧，优先提示未关联或短轨迹帧，并定期抽取无检测帧检查漏检。人工复核完成后运行：

```powershell
.\person_tracking\run.ps1 finalize-hard-negatives --dataset person_tracking/data/person_v5_d01_hard_review_20261106 --audit-out person_tracking/runs/audits/person_v5_d01_hard_20261106_r3 --export-out person_tracking/data/person_reviewed_v5_d01_hard_20261106_r3
```

审计会验证锁定基线标签/图像未改变、test 集仍一致、新增样本只进入既有 train 时间组。由于 v5 未达到指标目标，已从 v5 best 在 15:57:20 起的同一 train 时间组推理 120 秒，并为 D02/D03/D04 准备 v6 人工复核集：基线 856 张锁定，新增 35 张待复核（D02=12、D03=12、D04=11），全部进入 train，val/test 仍为 117/123 且未改变。来源推理和逐文件审计记录位于 `runs/v5_hard_review_source_cameras_20261106/` 与 `data/person_v6_hard_review_20261106/provenance.json`。

复核页已通过本机 `http://127.0.0.1:8770` 提供。候选的机器框仅供检查；用户需核对整幅图像，只保留真实人员框，无人图像确认空标签。完成后执行审计与导出：

```powershell
.\person_tracking\run.ps1 finalize-hard-negatives --dataset person_tracking/data/person_v6_hard_review_20261106 --audit-out person_tracking/runs/audits/person_v6_hard_20261106 --export-out person_tracking/data/person_reviewed_v6_hard_20261106
```

不要将当前 val 样本并入训练后仍声称同一 val 是独立评估。单纯提高 conf 会同时压低召回率，不能作为完整修复。

嵌套框过滤已在 v5 best 的 D01 15:57:20–15:59:20 片段实跑 1,200 帧：原始检测 1,995 个，抑制 305 个高包含度小框，保留 1,690 个检测，创建 18 条轨迹。该计数验证过滤路径实际生效，不等同于人工准确率或轨迹 IDF1；逐帧结果见 `runs/d01_hard_review_source_v5_20261106/D01/frames.jsonl`。ROI 仍只输出 `inside_roi` 标记，人员检测覆盖整幅画面。

当前单类数据集只支持 `person` 标签。Precision 不足时，硬负样本按整帧复核：有人就标全所有真实人员，无人就保存空标签，吊钩/钢丝绳等区域留空；YOLO 会把未标注区域作为背景学习，不能把它们画成 `person` 负框。Recall 不足时要补标漏掉的真实人员。若业务确实要求在图上框出干扰物，应新增 `distractor` 等正类别并一致标注，在多类别模型推理时只保留 `person` 输出；这需要扩展当前标注类别与训练/评估配置。

1. 检测验证：人工框真值，分通道报告 precision/recall、mAP50/50-95；另看小目标、俯视、遮挡、空场误报。
2. 跟踪验证：保留连续原视频片段，人工标注同一相机内人员 ID，以相同取帧方式计算 IDF1、HOTA、ID switches、丢失/重捕时间。独立静帧不能验证持续跟踪。
3. 工况验证：分别检查大车静止/移动、小车移动、震动、多人交叉、吊物经过人员前方、长遮挡后出现、视频断流/重连。框位置与 ID 均需回放核验。
4. 性能验证：Windows CUDA 的演示速度不代表 RK3588。盒端需替换检测后端为经校验的 RKNN 推理，保留跟踪/日志接口，并重新测量四路解码、预处理、NPU、后处理、跟踪、录像及长期热稳定性。

持续跟踪所需的连续人员框、人工身份 ID、遮挡规则、10 fps 帧映射及 MOT 格式说明见 [人员持续跟踪数据标注指南](人员持续跟踪数据标注指南.md)。现有静态复核网页不含轨迹标注功能；建议采用支持 Track 模式的视频标注工具，后续对接评估。

“吊物运动时区域有人”“人与工件都在移动”“人员伴随吊物移动”所需的时间信息不同，按业务选择逐帧检测或短时对应，详见 [检测与跟踪选型说明](人员与工件同时移动_检测与跟踪选型说明.md)。完整 MOT 标注主要用于系统性跟踪评估，不是所有业务启动前都需要完成的步骤。

## 6. 安装和限制

```powershell
# 已完成部署的本机通常不必再次安装
.\person_tracking\setup_env.ps1

# 另一台没有 NVIDIA GPU 的 Windows 机器
.\person_tracking\setup_env.ps1 -CpuOnly

# 回归检查
& 'D:\Anaconda\envs\crane-person\python.exe' -m pytest person_tracking/tests -q
```

依赖版本见 `requirements.txt`，实际环境冻结记录见 `environment-freeze.txt`。CUDA 版 PyTorch 自带运行库，不要求另装完整 CUDA Toolkit，但需要可用 NVIDIA 驱动。

RTSP：把 `cameras.json` 中 `source` 替换为相应主码流或第三码流地址。直播使用后台解码、仅保留最新帧、断流自动重连；重连和较长输入空档会建立新 session。直播日志时间是本机收到帧的时间，不是相机曝光时间；本次只有本地录像可验证，现场 RTSP 重连和四路实时吞吐仍须联调。

带框 MP4 使用固定分析帧率生成预览。直播实际处理速度不足、丢帧时，预览播放速度可能与真实时间不同，应以 JSONL 时间字段为准；原始录像仍应单独保存。

公开预训练检测不能保证识别所有现场人员；这套代码提供基础检测、图像轨迹及数据准备，不直接触发起重机安全控制或宣称满足全部十项业务验收。

模型与框架来源：[Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/)、[Ultralytics Tracking](https://docs.ultralytics.com/modes/track/)、[许可说明](https://www.ultralytics.com/license)。Ultralytics 代码和模型涉及 AGPL-3.0 或商业授权，产品交付应选择符合实际分发方式的许可。

## 人员遮挡分组验证（2026-11-07）

新增 `occlusion plan/sample/review/status/freeze/evaluate`，支持按遮挡比例、吊物/结构遮挡及远处小目标分别报告 Recall。D02/D03 的 180 张图像已全部复核，含 5 个人帧，真值冻结到 `evaluation/occlusion_20261107/frozen_v1/`。v7/v8 首次评估分别得到 P/R=36.36%/80.00%、33.33%/40.00%；其后以 v9 诊断得到 80.00%/80.00%（TP/FP/FN=4/1/1）。此集合先前已用于比较 v7/v8，故 v9 结果属于开发诊断，不能作为未触碰的独立验收。正样本来自同一人连续五帧，缺吊物及重度遮挡，也缺其他日期录像；这些缺项不能声称达标。详见 [人员遮挡分组验证结果](人员遮挡分组验证结果_2026-11-07.md)。

标注查看入口：<http://127.0.0.1:8773/?status=all>。页面不预填模型框，逐人填写遮挡/远近属性；如需改标，修改源标签并另建冻结版本与结果目录，保留本轮 v1。采样内容检查、标注规则、完整命令和覆盖限制见 [人员遮挡分组验证方案与操作说明](人员遮挡分组验证方案与操作说明_2026-11-07.md)。

当时的后续选样已把 `evaluation/` 下已完成的留出比较计入历史排除，避免将这 180 张及其邻近时段再次当作独立数据。四段原片严格时间组已无剩余；同日 30 秒保护规则下仅余 D03 约 10 秒，顺序采出的 10 张初看没有明确人员，保持待标且不计入本次结果。该批遮挡评估本身没有启动训练；之后基于重新标注数据另行完成 v9 训练，见 [v9 重标训练与验证记录](YOLO11n人员重标v9训练记录_2026-11-07.md)。仍需其他日期/作业的遮挡正样本。

## 原人员图片重新标注（2026-11-07）

按用户要求，基于当前 v8 人工数据建立 `data/person_v9_positive_relabel_20261107/`：**363 张已有人员框的图片已重新复核，518 张已确认无人图保留原标签及人工记录**。18 个标签变化经审计后导出 v9，训练已完成。复核口径、原始审计和分集数量见 [人员正样本重新标注说明](人员正样本重新标注说明_2026-11-07.md) 与 [v9 训练记录](YOLO11n人员重标v9训练记录_2026-11-07.md)。

用户随后指出 D03 16:20–16:23、16:25–16:26 的便装人员片段，要求补标并做专项训练。新增 73 张已人工标注并完成 v10：以 v9 best 初始化，只训练新增图片，第 19 轮早停，第 4 轮最佳。新增训练图片 P/R 从 63.64%/33.02% 提升到 88.35%/85.85%，属于训练拟合；原 655 张训练图片 Recall 从 91.28% 降至 87.60%，保留 46 张 val 的固定阈值 P/R 从 100%/91.67% 降至 62.50%/83.33%，存在旧场景退化。该片段原属 val 的共享时间组已整体退出本轮 val；test 保持封存。专项权重单独保存，运行配置未替换。完整结果和限制见 [v10 便装专项训练记录](YOLO11n_D03便装专项v10训练记录_2026-11-07.md)，准备阶段记录见 [D03 便装候选时段与分集检查](D03便装候选时段与分集检查_2026-11-07.md)。

用户确认后，v11 平衡训练已完成：保留原 655 张 train，新 73 张各占 3 个采样槽，共 728 张唯一训练图、874 个训练槽；保留同一 46 张 val 与封存 test。以 v9 best 初始化，基础与 bias 预热学习率均为 0.0001。第 24 轮早停，第 9 轮最佳；同一 val 的项目复评 mAP50/mAP50–95 为 95.67%/53.07%，固定 conf=0.10 的 P/R 为 73.33%/91.67%。新 73 张拟合为 97.12%/95.28%，原 655 张保持诊断为 79.45%/94.38%，但遮挡开发集退化到 2/5 检出（66.67%/40%），不能视为全面替换成功。完整证据、失败帧和后续遮挡补强方案见 [v11 平衡训练记录](YOLO11n人员平衡训练v11记录_2026-11-07.md)。

公开预训练 YOLO11n 对 D03 的基准全片识别也已完成，位于 `runs/d03_baseline_yolo11n_full_20261107/`。与 v9 的全部 41,213 个源帧和 PTS 对齐，参数一致；带框回放首尾可读，覆盖同样约 68 分 41 秒的可解码范围。计数与源片尾部限制见 [D03 基准模型识别记录](D03基准模型识别记录_2026-11-07.md)。

## 多场景遮挡、边缘截断与俯视补样（v12，2026-11-07）

已准备 `data/person_v12_occlusion_multievent_review_20261107/`，共 **135 张新增全帧待标图片**：D01 14、D02 44、D03 40、D04 37。人工复核后保留 132 张、排除 3 张，保留样本含 127 张人员图、5 张明确无人图和 202 个人员框。四路顺序解码、历史去重和逐图检查后，保留人员通行、多人作业、工件/吊具/结构附近人员、边缘截断及俯视等候选；所有图片保留 1920×1080，不按 ROI 裁图。页面不预填机器框，旧的已确认无人图无需重标。

新标注入口：<http://127.0.0.1:8776/>。画框规则、来源与限制、重启命令见 [v12 补样与标注说明](人员遮挡_边缘截断_俯视补样与标注说明_v12_2026-11-07.md)。冻结副本为 `data/person_v12_occlusion_multievent_frozen_20261108/`，合并训练候选集为 `data/person_reviewed_v12_occlusion_augmented_20261108/`：train 1006 个槽位、val 46、test 111，其中 train 唯一图像 860 张。v11 val/test 指纹保持不变，尚未启动训练或替换运行权重。采样沿用当前训练时间组并隔离已有开发片段；样本仍来自同日四段旧录像，不作为新的独立测试集。

v12 已完成训练：以 v11 best 初始化，YOLO11n、1280、batch=2、AdamW `lr0=0.0001`，第 21 轮早停，第 6 轮为 mAP 最优。训练器 mAP50/mAP50–95 为 97.50%/56.86%；使用固定 `conf=0.10`、匹配 IoU=0.50 在同一 46 张 val 复评得到 Precision 85.71%、Recall 100.00%（TP/FP/FN=12/2/0）。该 val 只有 12 个真值框且没有 D02/D03，test 仍未评估。完整记录见 [v12 训练记录](YOLO11n人员遮挡补强v12训练记录_2026-11-08.md)，最佳权重尚未替换运行配置。

## 7. H.264 参考帧警告

`reference picture missing during reorder`、`Missing reference picture` 和 `mmco: unref short failure` 来自底层 H.264 解码器，表示参考图像缺失或参考帧状态异常，不是 YOLO 权重加载或 CUDA 错误。随机跳转、录像缺少完整关键帧、码流损坏和直播丢包等都可能引起，不能只凭日志确定唯一原因。

本项目已在 D01 随机跳转时复现；从头解码的对照没有出现同类警告。还发现 D01 的实际 PTS 与“累计帧号 ÷ 25”存在差异，因此本次同时修正了读取方式和计时。没有关闭 FFmpeg 日志，也没有修改原片。直接重跑原命令即可采用新的默认方式：

```powershell
.\person_tracking\run.ps1 track --cameras D01 --start-clock 15:57:20 --duration 30 --decode-mode sequential --out person_tracking/runs/d01_sequential_demo
```

若从头顺序解码仍持续报警，或出现花屏、拖影、停帧，应保留原录像和 stderr，核查录像完整性、导出方式及原始相机码流；不要把损坏帧纳入标注。顺序解码可以避免本次的跳转触发条件，不能补回原片本已丢失的参考图像。详细证据见 [H.264 排查与修复记录](H264解码警告排查与修复_2026-11-05.md)。
