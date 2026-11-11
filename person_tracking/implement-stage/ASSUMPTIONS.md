# 实现假设台账

<!-- ASK mode: never; effort: balanced; assurance: draft -->

| ID | 未明确之处 | 本次选择 | Class | Source |
|---|---|---|---|---|
| A-001 | 本次是否开始训练 | 沿用此前“不训练”的要求，运行公开预训练 YOLO11n；视频采样和模型候选框不是人工真值，不据此声称检测准确率。 | semantic | user |
| A-002 | 持续跟踪的身份范围 | 每路独立 BoT-SORT，启用背景光流运动补偿，不启用 ReID；ID 仅在本次运行、同一路和同一连续片段内有效，长遮挡/出画重入可能换号。 | semantic | default |
| A-003 | 视频数据划分 | 根据文件名起始时刻及帧时间，把四路同一全局十分钟时段分入同一集合；边界各留 60 秒空档。时间只是初步对齐，同一天同一作业场景不能证明泛化能力。 | semantic | default |
| A-004 | 图像位置的含义 | 轨迹坐标为原始图像像素；框底边中心仅作位置近似，不表示真实鞋底接地点、地面距离或三维坐标。 | semantic | default |
| A-005 | 模型与依赖 | Ultralytics YOLO11n（仅 person 类）+ BoT-SORT；新建 conda 环境 crane-person，优先 CUDA，CPU 可运行。Ultralytics 的 AGPL-3.0/商业授权条件需在产品交付时按官方许可处理。 | interface | default |
| A-006 | 输入和输出接口 | 读取项目四视频，也允许配置 RTSP；每帧 JSONL 包含 source_frame、video_time_s、camera_id、session_id、像素框和 track_id；轨迹 ID 不跨源共用。 | interface | default |
| A-007 | 无人样本与标签审核 | 所有模型输出先存 proposals；未复核样本不能作为负样本。导出只接受已人工确认的标签，空标签必须明确确认无人。 | semantic | default |
| A-008 | 跳帧、丢失与计时 | 按源时间取帧，目标约 10 fps；跟踪丢失缓冲默认 2 秒、按设定分析帧率折算，并按时间再次检查。仅输出当前帧观测到的轨迹，不把预测轨迹伪装成观测。 | semantic | default |
| A-009 | H.264 跳转警告与视频计时 | 本地跟踪、采样默认从头顺序解码，按实际视频 PTS 选帧；缺失或非递增 PTS 才按标称 FPS 递推并显式标记。快速 seek 为可选项，seek 后帧号只作位置估计；不隐藏解码日志，不覆盖历史数据。 | semantic | observed-video |

## 备注

- 镜头运动补偿只是图像配准，不建立米制坐标，也不保证在强遮挡、反光、广角畸变下有效。
- 程序默认不配置业务 ROI，以免把尚未取得的实际 ROI 边界写成事实。
- 不增加长期身份识别模型；后续应通过人工连续序列标注验证 IDF1/HOTA 和换号率。
- `SWEEP_UNAVAILABLE`：本次没有可用的独立跨模型审查接口；不以本实现者自查冒充独立审查。
