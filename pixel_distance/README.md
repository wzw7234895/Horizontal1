# 固定镜头图像两点测距

## 对外调用接口

提供进程内 Python 接口和 HTTP/JSON 接口。两种方式均接收 D01-D04 四路同秒图像及参考图 A/B 像素；四路参与视觉候选，米制距离只使用组内相机对。接口字段、返回 JSON、错误语义、配置更新和启动命令见[对外接口说明](对外接口说明.md)。安装 HTTP 可选依赖：`python -m pip install -r pixel_distance/requirements-api.txt`。

**当前默认方案（2026-11-09）：**D01/D03 随大车、D02/D04 随小车；只有各组内部的安装几何固定。`video-distance` 每路按目标秒取一帧，四路都参与视觉候选搜索；米制三角测量分别在 D01/D03、D02/D04 组内完成。两点均在同组定位时可求距离，即使初始选点来自另一组；两点分属两组时，仅在已验证竖直高度基准下求高度差，不计算跨组三维距离。无可靠对应时默认返回 `unknown`，不假定两点在地面。见 [当前组内测距与跨组高度方案](当前组内测距与跨组高度方案.md)。下文保留的早期四机刚性、跨组车长诊断和单目结果属于历史试算。

本工具是**相机几何测量程序**，不进行 YOLO 训练，也不把像素长度乘一个全图固定比例。输入图片和两个像素后，在下列几何条件之一成立时可输出米制距离：

1. **同一地面平面**：两点确实位于已测绘的地面，使用单相机的去畸变像素到地面坐标的单应映射。
2. **任意高度的三维点**：A、B 各自是可在至少两路同步相机中确认的同一物理点，且每路相机在该帧有有效的三维位姿，使用视线交会求三维距离。
3. **预设参数估算**：暂按设计范围内的相机名义值，把两条像素射线投到同一假设水平面，直接给出未标定的近似距离。适合快速试算，不代表现场实测精度。
4. **实验性四路视觉候选与组内三维测量**：在一张参考图点 A、B，程序到另外三路寻找对应点，但只用同一固定安装组内的相机对三角测量。组内参数目前仍为未验收名义值；已知尺寸参照物用于离线标定与核验，不是运行时必需输入。

单张照片中的两点若在不同高度或未知深度，单靠固定焦距、安装高度及拍摄角度**无法唯一确定三维距离**。将悬空吊物下缘当成地面点会产生系统偏差。程序要求明确选择 `--surface ground`，并对图像尺寸、地面控制点覆盖范围和独立校验点误差做检查。

优化过程、现场证据、失败原因和后续验收顺序见 [`距离计算优化过程.md`](距离计算优化过程.md)。

## 现场依据与选型

- [设计说明](../检测能力体系详细设计说明（内部参考版）%20(2).docx)中的镜头为安装时设定的光学变焦，当前实际焦距无法由管理软件读取；焦距/视场的产品范围与相机标称安装高度均不是某台相机的可用内外参。正式机位将再调整，必须在最终安装、焦距、分辨率、裁剪状态下逐路标定。
- 四路试验视频实际为 1920×1080，与设计中的正式码流规格不同。D01/D03 是斜视人员视角，D02/D04 是近俯视吊点视角。近俯视更便于在无遮挡地面测足点和地面尺寸，斜视可补充悬空点三维视差。黄线和绿色轨道只是可寻找地面控制点的位置线索，其线宽、轨距未提供实测值，不能预设为米制尺。用户已确认亮黄色为顶部投射的真实光照，不能当作物理标尺或一律忽略。
- 相机安装在运动的大车或小车上，“镜头相对机架固定”不等于“相机相对车间地面固定”。**固定相机对的相对内外参可以跨画面复用**：对同一物理点在两路画面的对应像素三角化，两点间的距离不受整对相机一起平移或旋转影响。当前 `nominal-3d` 正是复用配置中的固定相对关系；其参数只是试算值。实测地面标定及需要绝对车间坐标的 `pose-state` 另绑定原始图片 SHA-256，跨帧须更新它们的世界位姿。
- [解决方案](../解决方案.docx)中预设四机间距/固定位置和“任意像素映射三维”的结论，仅在真实相机位姿、同步、同名点和深度条件满足时才成立。当前不依赖其未验证的滑块编码器或静态相机假设。

《解决方案》对相对运动的跨组相机提供了 `slider_x` 和 `rail_y` 的运行时输入路径；当前方案明确不估计这段运动。默认 `all_cameras_rigid=false`，`auto-3d` 允许四路视觉候选，但仅使用组内相机对做米制三角测量；`--trolley-origin-m` 在该模式下拒绝。解决方案中把像素投到 `Z=0` 地面的示例不能直接用于悬空物体三维点。已知尺寸参照物用于离线组内尺度标定和误差核验，不进入每秒测距输入。

## 环境

在项目根目录用已有环境运行：

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance --help
```

核心测距依赖 `opencv-python`、`numpy`；测试还需 `pytest`。`--matcher auto/loftr` 额外需要 `kornia`、PyTorch 和权重，边缘盒默认不启用。显式 `--matcher xfeat` 使用本地 XFeat 官方源码与权重，必须先设置 `PIXEL_DISTANCE_XFEAT_REPO` 和 `PIXEL_DISTANCE_XFEAT_WEIGHTS` 为本机目录和文件，运行时不联网下载。当前开发机的离线模型缓存位于 `D:\w\codex-conversations\2026-11-07\crane-pixel-distance\model-cache\hub\verlab_accelerated_features_main`；部署时须在边缘盒提供对应文件，并在目标硬件上实测耗时与内存。若改用新环境，可执行 `python -m pip install -r pixel_distance/requirements.txt`；该文件包含可选的较重依赖。

## B7：挂载重物时大车和小车同步运行

完整的运动分解、组内一致性检查、适用边界和现场优化路线见 [`B7_相对运动检测方案与优化路线.md`](B7_相对运动检测方案与优化路线.md)。

外部模块把挂载状态以 `loaded=True/False/None` 传给 `B7MotionDetector.update`；本模块不识别吊物。D01/D03 随大车运动，D02/D04 同时受到大车和小车运动影响。程序从每路相邻画面的静态地面候选特征估计相机位移，对两组相机分别取一致的运动，再用后一组减去前一组得到小车相对大车的运动。仅当两轴均超过速度阈值且方向连续一致至少 3 秒时，返回 `alarm_candidate`。任一路无法可靠估计、同组相机矛盾或相邻帧断档时，不用该区间确认告警。B7 结果不更新测距模块的相机相对位姿。

离线视频命令以 `--time` 为中心读取前后各 3 秒的四路画面，默认每 0.2 秒采样一组。`--load-state` 必须由外部挂载判断提供；`confirmed` 才分析运动。下例在 `Horizontal1` 目录运行，视频在父目录：

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance b7-motion `
  --time 2026-10-20T15:57:32 --videos-dir .. `
  --load-state confirmed `
  --out 'pixel_distance\demo_output\b7_155732.json'
```

实时接入时每次传入同一时间的四路 BGR 原始帧和单调递增的秒时间戳，不必缓存整个视频：

```python
from pixel_distance.b7_motion import B7MotionDetector
from pixel_distance.geometry import load_json

detector = B7MotionDetector(load_json("pixel_distance/camera_presets.json"))
result = detector.update(timestamp_s, {"D01": frame1, "D02": frame2,
                                       "D03": frame3, "D04": frame4}, loaded=True)
```

接口只保留上一组原始帧和最近 6 秒的区间摘要。`--sample-step-s`、`--min-motion-duration-s`、`--min-speed-mps`、`--max-group-disagreement-mps` 可调整采样、持续时间和运动门槛。`camera_presets.json` 的 `b7_axes` 暂设大车沿 Y 轴、小车沿 X 轴：D01/D03 在 X 轴方向相隔约 10 米，小车沿这一方向运行；现场需按真实安装方向和相机参数校正。现有速度为名义几何下的估算值，尚无现场挂载状态真值或同步运行标注，`alarm_candidate` 不能视为已验收的生产告警。边缘盒还需实测四路解码、光流和内存开销。

## 直接用预设参数估算

四路配置在 [`camera_presets.json`](camera_presets.json)，供命令行 `--config` 和 Python API 读取，可直接修改。D01/D03 对应斜视人员相机，D02/D04 对应近俯视吊点相机；具体映射沿用试验视频标签。`height_m=10` 来自设计说明的 8~12 米范围的中间值。水平视场角 D01/D03 暂取 60°、D02/D04 暂取 80°，均在说明书产品范围内，但**不是实际变焦测量值**。俯角分别暂设 45°、85°，只是根据斜视/近俯视画面的初始假设，说明书没有给出角度数值。主点暂取画面中心、像素暂按正方形、镜头畸变暂忽略。这些参数均可在 JSON 中更改，尤其广角画面边缘的畸变可能显著影响结果。

两个物理点必须经用户确认都落在**同一水平面**，因此命令行要求显式添加 `--assume-same-plane`。默认高度为地面 `0`；若确实同处距地 2 米的水平面，使用 `--plane-height-m 2`。若两点在不同高度或高度未知，单帧模式不能求真实三维距离，应使用下面的多相机模式。

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance estimate --image 'E:\images\D04-frame.png' --camera D04 --assume-same-plane --out 'E:\results\estimate.json' --annotated 'E:\results\estimate.png'
```

不提供坐标时会弹窗依次点 A、B；也可以直接加 `--point-a 1400,700 --point-b 1600,700`。缩放显示会自动还原原图坐标。程序支持 16:9 的不同分辨率，但输入应为**同一相机完整画面**；等比例裁剪仍可能通过比例检查，必须由用户避免。输出中的 `status=nominal_estimate_not_calibrated` 表示近似估算，`parameters_used` 记录实际采用的参数；`height_only_sensitivity` 只展示保持其他参数不变时，说明书 8~12 米高度范围带来的距离变化，**不是总误差或置信区间**。

终端默认只显示中文点位、距离和计算假设。需要完整参数时，使用 `--out` 保存 JSON 文件，或加 `--json` 在终端打印完整结果。

代码调用接口：

```python
from pixel_distance.geometry import load_json
from pixel_distance.preset import estimate_on_plane

config = load_json("pixel_distance/camera_presets.json")
result = estimate_on_plane(config, "D04", (1920, 1080), (1000, 500), (1200, 500))
print(result["distance_m"])
```

为改变相机参数，可编辑 JSON 中对应相机的 `height_m`、`horizontal_fov_deg`、`down_tilt_deg` 和 `roll_deg`；代码接口也可在调用前修改 `config["cameras"]["D04"]["height_m"]`。请优先用现场实际测量结果替换名义值。

## 自动跨视角三维候选（实验性）

### 从四路视频直接测距

`video-distance` 只要求指定时间和参考相机；在参考帧中依次点击 A、B，或通过 `--point-a/--point-b` 指定原始视频分辨率下的像素。程序在当前目录自动找到 `D01_YYYYMMDDHHMMSS.mp4` 至 `D04_YYYYMMDDHHMMSS.mp4`，用文件名起始时间和 [`camera_presets.json`](camera_presets.json) 的 `video_clock_offset_s` 每路抽取该秒一帧。四路均参与同名点候选搜索；组内两路通过几何检查后才提供米制结果。时间只接受到秒的 `YYYY-MM-DDTHH:MM:SS`；按当前业务约定忽略秒内帧差。

现场 H.264 文件按帧号跳转时可能产生大量 `reference picture missing`/`mmco` 解码警告。`video-distance` 现在在隔离进程中读取目标帧，终端不显示这些重复信息；结果 JSON 的 `video_frames.*.decoder_warning_lines` 保留警告行数，跳转或解码失败仍报错。不要直接改用 FFmpeg 容器秒数跳转：本次 D04 对相同偏移量得到的烧录时间比目标早约 10 秒。

默认 `--matcher auto` 会依次尝试 SIFT、预训练 LoFTR、双目校正后的极线图像块匹配。最后一级属于光学外观匹配，要求相机位姿与镜头模型足够准确；同一路视频的连续帧光流只能跟踪时间变化，不能单独解决大视角跨相机同名点。上述算法只在发起测距时运行，不用于逐帧实时告警；LoFTR 需要本地 PyTorch/Kornia 与模型权重，正式边缘盒部署前必须测量实际耗时与内存。显式 `--matcher sift` 可只运行轻量方法。匹配器更换不能修复错误的相机位姿，也不能保证无纹理、被遮挡或不同视角只见不同表面的任意像素有唯一同名点。

场景上下文匹配会在同组两路画面寻找选点附近具有空间一致性的特征，候选仍须通过极线和三角重投影检查。它不能补出遮挡、无纹理或不同视角不可见的物理点。旧版 15:57:32 的[约 4.124 米结果](demo_output/video_distance_场景上下文复核_155732.json)是单目假定地面的历史试算；当前默认会返回未知，只有显式启用同平面试算才会给出这类数值。匹配器及解码耗时仍须在边缘盒实测。

```powershell
Set-Location 'E:\横向项目-目标检测'
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance video-distance `
  --time 2026-10-20T15:57:32 --reference D04 `
  --point-a 624,513 --point-b 647,535 `
  --out 'pixel_distance\demo_output\video_distance_155732.json' `
  --frames-out 'pixel_distance\demo_output\video_distance_155732_frames'
```

若不指定 A/B 坐标，会弹出抽取的参考画面供点击。A、B 均在同组另一相机通过对应检查后，才输出组内三维候选；失败默认 `distance_m=null`。只有显式添加 `--assume-same-plane`，才启用原有的单目水平面试算，可用 `--fallback-plane-height-m` 设置已知平面高度。单目试算不恢复悬空点深度，也不供告警使用。

### 地面候选分割（可选）

`segment-ground` 使用本地 SegFormer ADE20K 的 `floor` 类别输出绿色候选叠加图，也可诊断参考图的 A/B。模型不是为本车间和俯视镜头训练的，绿色仅表示模型候选，不能证明所见表面位于地面或给出其高度。地面分割只处理参考画面，不会替代跨视角同名点或三角测量。当前开发机模型目录是 `D:\w\codex-conversations\2026-11-07\crane-ground-segmentation\segformer-b0-ade20k`；换机器时应提供本地 `config.json`、`preprocessor_config.json` 和权重，环境需要 PyTorch 与 `transformers==4.46.3`。

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance segment-ground `
  --image 'pixel_distance\demo_output\video_distance_155732_frames\D04_20261020_155732.png' `
  --model 'D:\w\codex-conversations\2026-11-07\crane-ground-segmentation\segformer-b0-ade20k' `
  --point-a 1278.17,445.71 --point-b 1254.86,922.97 `
  --overlay 'pixel_distance\demo_output\ground_segmentation\D04_overlay.png' `
  --out 'pixel_distance\demo_output\ground_segmentation\D04.json'
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance video-distance `
  --time '2026-10-20T15:57:32' --reference D04 `
  --point-a 1278.17,445.71 --point-b 1254.86,922.97 --matcher sift `
  --ground-model 'D:\w\codex-conversations\2026-11-07\crane-ground-segmentation\segformer-b0-ade20k' `
  --out 'pixel_distance\demo_output\ground_segmentation\D04_video_distance.json'
```

启用 `--ground-model --assume-same-plane` 后，只有组内三维失败且 A/B 都达到 `--ground-threshold`（默认 0.75）及局部支持条件时，才允许假定地面单目试算；否则 `distance_m` 和 `height_difference_m` 均为 `null`。达到条件仍只是条件估算，不是已验证的地面真值。`--ground-model` 不能与非零的 `--fallback-plane-height-m` 同用。未加 `--assume-same-plane` 时不会单目回退。

15:57:32 四路画面的候选像素占比为 D01 39.7%、D02 0.0%、D03 17.0%、D04 2.9%；[叠加图和 JSON](demo_output/ground_segmentation)显示 D01 部分车辆表面被误分，D02/D04 大面积真实地面被漏分。D04 示例 A/B 都被模型标成 `wall`，因此带分割门禁的测距结果为 `unknown`，并未得到真实三维距离或高度差。当前模型只能用于离线诊断和保守门禁，不能直接承担现场地面判定。边缘盒部署前需采集现场标注样本验证各路误分/漏分与推理耗时、内存；必要时换经现场验证的轻量分割模型。

如果两个点都通过多视角三角测量，JSON 还给出 `height_difference_m`，定义为 B 的世界坐标 Z 减 A 的 Z；正值表示 B 较高。当前名义外参未标定，该值仍只是条件估算。若退回单目，字段为 `null`，终端明确显示“高度差无法确定”。

### RoMa 稠密匹配离线诊断

[RoMa 官方实现](https://github.com/Parskatt/RoMa)可对任意查询像素提出跨视角候选，`roma_probe` 使用完整 outdoor 权重，对 A/B 各自做前向映射和反向回查，输出置信度、回查误差与名义极线误差。它**不向正式测距命令注入未经验证的匹配点**，因此不会把低置信候选变成假高度。安装 `romatch` 后，可在开发机 CUDA 环境运行：

也可在四路视频命令中显式加 `--matcher roma`；它只把同时通过前向/反向置信度、3 像素回查和 3 像素名义极线检查的点交给三角测量。当前模型权重需通过 `TORCH_HOME` 指向已下载目录，未配置时 `romatch` 可能自行下载。它适合离线或按需复核，不能视作已达到边缘盒实时预算。

未安装可选依赖时运行 `& 'D:\Anaconda\envs\crane-person\python.exe' -m pip install romatch`。模型默认会从官方地址下载权重；首次运行前先将 `TORCH_HOME` 指向 D 盘等有足够空间的目录。

```powershell
$env:TORCH_HOME = 'D:\w\codex-conversations\2026-11-07\pixel-distance-roma\torch'
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance.roma_probe `
  --reference D04 --reference-image 'pixel_distance\demo_output\video_distance_155732_frames\D04_20261020_155732.png' `
  --target D03 --target-image 'pixel_distance\demo_output\video_distance_155732_frames\D03_20261020_155732.png' `
  --point-a 1278.17,445.71 --point-b 1254.86,922.97 `
  --out 'D:\w\codex-conversations\2026-11-07\pixel-distance-roma\roma_probe_D04_D03.json'
```

完整 RoMa 权重约 425 MB，DINOv2 权重约 1.13 GB，开发机 RTX 3060 Laptop 6 GB 上模型加载后占用约 2 GB GPU 显存；这些不是边缘盒性能数据。15:57:32 的 D04 A/B 在 D03 的 448 分辨率诊断中，双向回查误差约 16/56 像素，名义极线误差约 51/34 像素；提高到 560 分辨率后回查误差约 130/322 像素。D01/D02 也未通过双向检查。说明当前图片可产生**候选位置**，尚不能确认两个所选像素均为同一物理点，更不能据此输出真实高度差。RoMa、[LightGlue](https://github.com/cvg/LightGlue)和[RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo)的适用性不同：前者稠密但资源较重，LightGlue 是稀疏特征匹配，RAFT-Stereo 需要已校正的双目输入；没有一种能保证被遮挡或无唯一纹理的任意像素匹配。

这次完整四路 `video-distance --matcher roma` 复核的结果保存于 `D:\w\codex-conversations\2026-11-07\pixel-distance-roma\video_distance_roma.json`：D01/D02/D03 对 A/B 均未通过视觉和当前名义几何检查，`height_difference_m=null`。要把候选用于米制高度，至少需现场标定每路内参/畸变与同一时刻的相对外参，并在有可辨识表面特征的点上验证双向匹配及三角测量误差；仅换模型不能使被遮挡或歧义点可观测。

当前 `video_clock_offset_s` 通过现场 15:57:32 截图反查四路视频得到，D01/D02/D03/D04 分别暂设 `-0.2/+4.8/-3.0/-4.0` 秒；它们是视频文件名时间轴与画面叠字的近似换算，并非设备时钟标定值。输出 JSON 的 `video_frames` 列出每路帧号、帧率和时间偏移；`--frames-out` 可保存用于目视核对的原始帧。跨视角候选的名义极线容差已按要求设为 `epipolar_candidate_threshold_px=20`，这只是进入局部匹配的门槛，不代表 20 像素误差内就属于同一物理点。

2026-11-07 红点复核：用户提供的 D01/D02/D03 截图是 1635x921，必须分别按 `1920/1635`、`1080/921` 换算为原视频坐标。D04 A/B 为 `(1278.17,445.71)`、`(1254.86,922.97)`；红点约为 D01 A/B `(1565,984)/(784,925)`，D02 `(683,100)/(663,465)`，D03 `(964,1000)/(434,1031)`。红点是近似指示，不能视为精确同名点。此前 SIFT/LoFTR/XFeat 的整幅图遮罩排除了顶部 12% 和底部 10%，恰好覆盖这些红点中的多数；现已改为仅过滤黑区和最外侧 0.5% 边框。修正后现场帧的自动 SIFT/LoFTR/极线块仍未确认 A/B，详情见 [`D04_红点复核_自动匹配遮罩修正后.json`](demo_output/D04_红点复核_自动匹配遮罩修正后.json)。

现有 RoMa 候选与红点在 D03 相距约 32/5 像素，在 D01 相距约 416/506 像素，在 D02 相距约 636/807 像素。以当前名义位姿计算，三路红点 A/B 的极线误差分别约 D01 `473/808`、D02 `739/97`、D03 `48/35` 像素；D03 的两点也不能通过 3 像素三角重投影验证。D04/D03 四个可见车板角标给出的**车板平面**映射预测 D03 A/B 约 `(961,980)/(418,990)`，与用户 D03 红点约差 20/44 像素，见[对照图](demo_output/D03_红点与车板平面预测对照.png)。这是有平面归属前提的候选修正，不能用于悬空工件、人员或任意高度点，也不能用红点差异反推可靠的米制高度。D01/D02 红点有落在吊装结构上的可能，需先证明同一物理表面可见。对 D04/D03 的 RoMa 稠密结果直接拟合极线矩阵虽有约 1700 个拟合内点，但四个留出车板角标中有两个仍偏约 92/118 像素，因此不能用该矩阵替换已标定外参。要做通用跨视角测距，需以独立可见三维点核验镜头内参、畸变和相对位姿，并在运行时坚持双向匹配、极线、正深度、重投影与多视角三维一致性检查；未通过时高度差保持未知。

对 D04 车板区域另外抽取 75 个网格点比较 RoMa 与四角标记单应映射，RoMa 置信度中位数约 `0.06`，位置偏差中位数约 `33` 像素，只有 `24%` 落在 20 像素内。该平面映射是局部几何参照，不是这些网格点的逐点实测真值；检查结果表明当前 RoMa 不能直接可靠转移该区域内的任意像素。当前程序仍需在严格验证未通过时拒绝三维输出。

15:57:32 的原始视频帧为 1920x1080，和汇报截图的 1784x1003 不可混用像素坐标。以 D04 平板内部代表点 `(1240,550)`、`(1300,800)` 测试，SIFT/LoFTR 在 D01/D02/D03 都未确认两个点，返回的约 2.208 米仍是**单目假定地面**距离，不能当作平板实测。D04/D03 两幅图中车体四个红色标记可以自动识别，但按当前名义相机位姿，它们的极线误差约为 `57.7/30.8/15.5/13.6` 像素，已有三个超过 20 像素门槛；临时把门槛放宽到 100 像素后，SIFT 和 LoFTR 仍没有在这两个选点附近找到可靠的局部对应。当前点击的 `(1278,446)`、`(1255,923)` 用双目校正图像块复核时，D04→D03 两点均落在有效校正画面外，D02 的最高相似度仅 `0.69/0.62`，D01 则为纹理不足或边界无效。可见这次同时涉及名义位姿不一致和局部纹理不足。四标记可以对**该车平板平面**建立像素单应关系，但不能把这层关系用于工件、人员或其他高度的点，也不能凭它替代相机间米制位姿。

以 D04 原视频的 `(624,513)`、`(647,535)` 运行上述命令，当前 SIFT 分别在 D01/D02/D03 得到 `0/12`、`1/30`、`0/13` 个名义极线候选，未确认两个所选点。工具返回约 **0.284 米的单目假定地面估算**；此数值不能用于声称悬空工件的真实边长。

### 直接运行名义三维估算

`nominal-3d` 直接接受两路图片和像素坐标；不要求先手写视角 JSON。相机光心与朝向在 [`camera_presets.json`](camera_presets.json) 的 `nominal_world_poses` 中开放修改，目前 D04 暂设 `(0,0,10)` 米、D03 暂设 `(8,1.28,10)` 米，视场角和俯角沿用同文件的相机参数，畸变暂按零处理。这些是可运行的试算值，不是现场测量值。

只给参考图 A/B 点时，程序自动找另一视角的同名点；当前现场画面对此尚未成功，会返回 `unknown`。若两路画面中 A/B 的同一物理位置可直接辨认，可用 `--target-point-a/b` 指定；或加 `--click-target`，仅在自动匹配失败时弹出另一视角供点击。两个点可处在不同高度，程序分别用两路视线三角化，不套地面映射。以下命令在当前截图上给出 **3.461 米的名义三维估算**：

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance nominal-3d --reference D04 --reference-image '汇报材料\现场同刻画面_2026-11-05\02_主钩吊点_15-57-32.png' --target D03 --target-image '汇报材料\现场同刻画面_2026-11-05\04_左侧人员侦测_15-57-32.png' --point-a 1205.4743,412.8686 --point-b 1200.3771,858.8686 --target-point-a 893.75,866.90 --target-point-b 402.85,871.87
```

替换两张图和四个像素坐标即可测另一组点；要自己点击参考图 A/B，省略 `--point-a/b`。数值的前提是两路 A/A、B/B 确为同一物理点且假定相机参数正确。设置任意参数并不能使估算自动成为真实距离；程序在输出中保留 `actual_distance_determined=false`。用 `--out` 可保存完整 JSON。

固定**组内**双目关系对应一个可复用的映射 `(u_A^1,v_A^1,u_A^2,v_A^2) -> XYZ_A`，B 点同理，再求 `|XYZ_A-XYZ_B|`；它不要求 A、B 在同一平面。参考图的新点可先视觉转移到另一组，但最终仍须在某一组的两路视角都找到同名像素。仅由参考图的两点像素间距不能唯一确定三维距离；跨组水平关系未知时不输出完整三维距离。

`camera_presets.json` 现为 `all_cameras_rigid=false`。`cameras.*.rig` 将 D01/D03 与 D02/D04 分成两组，`center_in_rig_m` 是各组内的固定光心位置；名义朝向暂取 `nominal_world_poses.*.azimuth_deg`。默认 `auto-3d` 只使用参考相机的同组视角。`nominal_world_poses.*.center_m` 是旧四机刚性试算字段，默认组内路径不使用。名义值均待现场校准。

用户确认 D01/D03 两台行车相机的横向组内间距至少 10 米；当前配置恰好用 10 米作名义值。这个下限**不是** D03 与小车相机 D04 在某一帧的距离。实际可确定的组内距离、相机高度/角度/朝向、分辨率和焦距须录入同一坐标约定；光圈影响清晰度和点位定位误差，不直接决定针孔成像的米制比例。

将四路同一时刻的原始全幅帧分别传入 `--d01` 至 `--d04`，在 `--reference` 指定的图中点击**任意两个需要测量的物理点** A、B。也可以直接给 `--point-a u,v --point-b u,v`；A/B 只是本次选点的名称，不限定为运输车或工件的固定位置。默认用 SIFT 建立视角关系，再检查局部对应、极线、正深度和三角测量重投影。只有显式指定 `--matcher auto` 或 `--matcher loftr` 才可能加载较重的预训练 LoFTR；首次运行需下载权重。它不训练本项目模型。

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance auto-3d --reference D04 --d01 'E:\frames\D01.png' --d02 'E:\frames\D02.png' --d03 'E:\frames\D03.png' --d04 'E:\frames\D04.png' --out 'E:\results\auto3d.json' --annotated-dir 'E:\results\matched'
```

终端只显示简短中文结论。`--out` 记录每一路候选点和未匹配原因；`--annotated-dir` 生成各路 A/B 标注图，必须目视核对是否为同一个物理点。若只在一对相机中通过检查，结果标为**三维候选距离**，没有独立视角交叉验证。画面中的黑色边框、顶部时间戳和底部标签会在匹配时屏蔽。低纹理地面、反光金属、遮挡或两个视角看到的是轮廓不同表面点时可能返回“无法可靠计算”。四路截图的秒级时间戳不证明曝光同步；运动物体需要帧级同步。

历史四机刚性试验曾对 D03/D04 使用固定名义位姿；该样本的 D03 有 14 组双向 SIFT 特征，其中 0 组通过名义位姿的 3 像素极线检查。当前默认 `auto-3d` 可在 D03/D04 之间寻找视觉候选，但不使用这条未知跨组基线三角测量；上文 `nominal-3d` 的结果只可作离线条件试算。

需要连续帧辅助判断时，增加 `--later-reference 'E:\frames\D04-next.png'`，传入**同一参考相机、同尺寸的相邻原始帧**。光流只跟踪本次选中的两个点，并用静态场景特征估计相机/背景的整体运动。详细 JSON 的 `temporal.points.A/B` 分别给出跟踪结果；“随背景运动”不证明该点在地面，“偏离背景运动”也可能是视差或跟踪错误，均不能直接推出离地高度。被点击的像素须是物体本身的可重复识别特征；点击物体旁的地面只能得到地面点的位置，无法据此推算物体边长。

连续帧路径将 1920×1080 图缩到最大边 960 像素、背景角点上限 800、选点轨迹固定为 2 条，默认 SIFT 路径不调用神经网络。参考帧 SIFT 特征在三个目标视角间只提取一次。当前开发机上 D04 相邻原始帧的连续帧分析约 66 毫秒，四路图像加连续帧的单次命令约 1.4 秒；这些数字**不是边缘盒实测帧率或实时性保证**。设计说明只给出约 26 TOPS 推理算力，尚无可确认的 CPU 型号和内存带宽。正式部署需在盒上测试解码、SIFT、光流、四路调度的端到端耗时和峰值内存；若超预算，可只在用户发起测量时运行跨视角匹配，并减少同时处理的相机对。整个方案的米制准确度仍取决于真实内参、相机间基线及运行时相对位姿；当前配置中的数值是可修改的试运行假设。

现有同刻截图的例子：D04 上两个绿色地面点 `(1211.85,410.32)`、`(1209.30,866.51)` 未能在其他视角可靠自动对应，程序返回 `unknown`；D04 工件邻近点 `(624,513)`、`(647,535)` 在 D02 找到对应，使用**假定 2 米基线**得到约 1.027 米的候选值。它未与现场真值比较，也不能作为工件实测长度。对应点的检查图在 `demo_output/site_auto3d_load/`。

用户标注图中 AB/CD 红点的历史同地面预设分别约为 4.183 米和 3.998 米。这仅说明名义地面投影的旧算法会产生接近的数值，不证明它们属于同一平面或为真实距离。旧复核曾尝试固定四机跨组试算；当前配置已关闭。四路可提出视觉候选，米制三角测量只使用 D04/D02 或 D01/D03 组内相机对。

为避免只用 AB/CD 个例判断通用能力，对四路同刻截图分别作为参考视角做了 50 至 52 个分布式采样点的 SIFT 匹配诊断，所有跨相机对均未得到可通过局部检查的采样点；D04 参考图使用 LoFTR 检查 52 点，仍未得到可通过检查的跨相机对应。相同 D04 相邻原始帧用 SIFT 可匹配其中 20/52 点，说明程序不是对任何图像都失败，但**时间匹配不提供双目基线和悬空点深度**。记录见 `demo_output/site_grid_coverage_sift_all_refs.json` 和 `demo_output/site_grid_coverage_loftr_all.json`。这是有限网格的可匹配性诊断，不是任意点的成功率，更不是米制精度验收。

### 手工对应点的条件试算

当自动跨视角匹配失败，但能手工在两路画面中指定同两个物理点时，可运行 `manual-nominal-3d`。输入 JSON 为每路提供图像、两个点的原图像素坐标、相机中心米制坐标与水平朝向；俯角和水平视场角读取 `--config`。每路可选 `intrinsics` 指向实际内参 JSON，以使用真实焦距、主点和去畸变结果。若不给内参，程序按名义视场角与图像中心建立针孔射线。程序检查两点视线交会角、正深度和视线最近间距，超出 `--max-ray-gap-m` 返回未知。`nominal-3d` 和 `manual-nominal-3d` 的默认上限为 **0.5 米**，可按需显式调整；该阈值**不是测距误差上界**，放宽后通过检查的结果仍只是名义参数下的条件估算。

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance manual-nominal-3d --views 'pixel_distance\demo_output\D04_D03_手工同名点_名义参数.json' --out 'pixel_distance\demo_output\D04_D03_名义三维结果.json'
```

本次 D04/D03 红点按“D04 的 A 对应 D03 右侧红点，B 对应左侧红点”试算。在**假定**两相机同高 10 米、D03 相对 D04 的水平位移 `(8, 1.28)` 米、两路朝向分别为 0°/90°、采用配置中 80°/60° 水平视场角和 85°/45° 俯角时，得到约 **3.46 米**；两点的视线最近间距约 **0.20/0.21 米**。但 D03/D04 的实际跨组位置未知，同一像素和角度若将水平位移改为 `(4, 0.64)` 米，仍能得到较小视线残差，距离变为约 **1.73 米**。这两个值都是条件试算，**不能据此确定真实距离或给出可信误差区间**。要得到可验收米数，应从同一时刻两刚性相机组的共同地面锚点、经标定的多视角地图或可靠的位置测量恢复跨组相对位姿，再对镜头畸变、点位误差、帧同步偏差和外参误差传播做验证。

用户补充：绿色轨道上的运输车车身前后总长约 **4–5 米**，两端在图中可辨认。如果本次两路 A/B 的确分别落在这辆车的同一前、后端点，则此范围可作离线尺度诊断。`manual-nominal-3d --known-length-range-m 4,5` 只报告是否相符，**不自动调整相机参数**。以后可以换用任意有独立已知长度、端点可确认的物体；其长度不应由同一测距结果反推。详见 [`运输车车长尺度核对_2026-11-07.md`](运输车车长尺度核对_2026-11-07.md)。

### 离线参照物诊断

`reference-check` 接受任何**独立获得长度范围**的参照物及其 A/B 测距结果，输出偏差范围与条件尺度比值，只用于定位焦距、位姿、对应点或同步问题。参照物可以是运输车，也可以是其他工件或固定结构；必须确认 A/B 对应的确是给定长度的物理端点。单段长度只能检查或约束尺度，不能独立求出两组相机完整相对位姿。诊断得到的比值不能直接写入运行时配置；应使用不同位置、方向和高度的参照物进行独立验证。

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance reference-check --measurement 'pixel_distance\demo_output\D04_D03_名义三维结果.json' --reference-name '运输车' --length-range-m 4,5 --out 'pixel_distance\demo_output\运输车离线长度核对.json'
```

本次运输车四个红色端部标记还可用 `vehicle-diagnostic` 做特例几何检查。旧输出中的约 **4.49 米**是先给定车长 4–5 米再换算出的条件数值，不是独立车长测量，也不能用于任意其他选点。正式 `auto-3d` 默认拒绝 `vehicle-diagnostic` 生成的位姿状态；`--allow-diagnostic-pose` 仅允许明确标记的离线试验。

### 运行时位姿与任意点

`init-pose-state` 和 `track-pose` 是保留的历史位姿状态接口；短程跟踪会累积漂移。当前 `auto-3d --pose-state` 对四路均尝试视觉匹配，但只使用固定组内相机对测距；状态绑定图像 SHA-256，避免拿旧状态测新帧。两点分属两组时可使用 `init-height-state` 的状态加 `--height-alignment` 自动尝试高度比较，或用 `compare-height` 输入人工确认的同名像素。不能把两组水平坐标当成同一个坐标系求完整距离。

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance init-pose-state --views 'E:\calibration\views.json' --ground-z-m 0 --out 'E:\calibration\pose-state.json'
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance track-pose --state 'E:\calibration\pose-state.json' --frames 'E:\frames\adjacent.json' --out 'E:\calibration\pose-state-next.json'
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance auto-3d --reference D04 --d04 'E:\frames\D04.png' --d03 'E:\frames\D03.png' --pose-state 'E:\calibration\pose-state-next.json' --point-a 624,513 --point-b 647,535 --out 'E:\results\auto-3d.json'
```

在已知两相机当前共享位姿时，可为 `auto-3d` 显式加 `--matcher patch`：程序先校正双目图，再沿每个选点的极线搜索局部图像块，并检查纹理、候选唯一性及双向一致性。这是无需训练的选点自动对应实验路径；两路原图须同尺寸，且所选物理点在另一视角可见。若校正后落在无效区域、被遮挡或图像块重复/无纹理，逐点返回 `unknown`。没有 `--pose-state` 时拒绝运行。合成双目测试通过；当前 D04/D03 吊物点使用车长诊断位姿仍未匹配，不能作为现场精度证明。

若业务允许约 1 秒的**结果延迟**，可缓冲各路画面并选曝光时间最接近的帧；结果延迟与两路曝光差是不同量。只保证叠字同秒时，应按实测最大相对速度和允许测距误差评估时间项，不能默认其影响可忽略。

这些命令中的 E 盘路径是待替换示例。当前仅复用固定组内几何，尚无正式内外参标定；自动匹配也不能保证确认任意新点。此前 D04/D03 跨组吊物点离线试验未通过极线与局部匹配检查，不能用车长诊断值替代。SIFT、XFeat、Tiny RoMa 的历史对照见[跨视角匹配现场复核](跨视角匹配现场复核_2026-11-07.md)。

### 违规事件画面

边缘盒形成的违规事件应保留：违规瞬间的**主码流视频帧**、以违规时间为中心的**第三码流前后各 5 秒片段**，以及设备和违规类型等事件元信息。空间测距优先使用各相机同一曝光时刻的主码流原始帧；第三码流片段用于前后轨迹、遮挡和对应点的时序复核。不同码流即使来自同一相机，也须核验时间偏移、裁剪、缩放和像素坐标映射；现有 `--later-reference` 只接受同尺寸原始帧，不能把第三码流帧直接当作主码流的后续帧。仅到秒的画面文字不足以证明四相机曝光同步。

相机移动且新旧画面仍有重叠时，可以复用已有结果，但要区分两种情况。若新旧画面跟踪到**同两个不动的物理点**，且旧距离已有可信三维测量，则两点距离本身不随相机移动而变化；只需验证两条轨迹及点的静止性，无须重新三角测量。若在重叠区域选的是**新的两个点**，地面点可在已知米制地面图上通过当前帧单应映射求距；任意高度的新点仍需当前帧多视角三角测量或已有三维地图点，不能仅凭画面重叠推断深度。若物体自身移动、遮挡、跟踪丢失，则旧点测距结果不能自动套用。

在当前“两组各自固定、组间水平运动未知”的规定下，**同组两点相对距离**可复用组内双目几何，不需要完整车间地图；新点可从另一组参考图视觉转移，但仍需在同一固定组的两路画面建立同名对应。跨组只可在离线竖直基准对齐后比较高度。历史短程位姿更新模块不参与默认测距，也没有持久米制地图或全局重定位；边缘盒上仍须实测解码、匹配与内存开销。

## 单相机地面测距

准备：在最终相机画面中的**同一块近似平整地面**布置并实测至少 6 个分散的控制点、另留至少 2 个独立校验点。每个点要有以米为单位的现场平面坐标；点位应覆盖将来点击的区域，不要集中在一条黄线。检查点须由测量人员独立提供，业务侧给出最大允许检查误差。若地面坡度/局部起伏超出容差，应分区建平面或改用三维模型。

先用不同位置、角度的至少 10 张棋盘格**原始全幅**照片标定当前相机实际内参与镜头畸变。`--corners` 是内角点列数、行数；`--square-m` 是现场实测单格边长。普通镜头选 `pinhole`，鱼眼镜头选 `fisheye`，应检查重投影残差并复核边缘棋盘格。已有可信厂家/现场内参时，也可按下面 JSON 格式提供；若没有内参，平面标定可省略 `--intrinsics`，但镜头畸变会进入系统误差，必须用独立检查点覆盖边缘区域验证。

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance calibrate-intrinsics --images 'E:\calibration\cam-D02\*.png' --corners 9,6 --square-m 0.025 --model pinhole --out 'E:\calibration\cam-D02\intrinsics.json'
```

`points.json` 格式如下；以下字符串是**待填实测值**，不是项目参数。`pixel` 可省略，由程序按 JSON 顺序弹窗点击；也可直接填写原图像素坐标。至少 6 个 `control` 和 2 个 `check`，每个 `id` 唯一。

```json
{
  "points": [
    {"id": "G01", "role": "control", "xy_m": ["实测X", "实测Y"], "pixel": ["原图u", "原图v"]},
    {"id": "G02", "role": "control", "xy_m": ["实测X", "实测Y"]},
    {"id": "V01", "role": "check", "xy_m": ["实测X", "实测Y"]}
  ]
}
```

省略项必须补齐，不能将上述占位字符串直接传给程序。`--max-check-error-m` 填业务批准的**独立检查点最大误差**，不是猜测的精度承诺。

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance calibrate-plane --image 'E:\images\frame.png' --points 'E:\calibration\cam-D02\points.json' --intrinsics 'E:\calibration\cam-D02\intrinsics.json' --max-check-error-m 0.05 --out 'E:\calibration\cam-D02\ground-frame.json'
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance measure --image 'E:\images\frame.png' --calibration 'E:\calibration\cam-D02\ground-frame.json' --surface ground --out 'E:\results\measure.json' --annotated 'E:\results\measure.png'
```

`measure` 默认弹窗依次点 A、B；也可传 `--point-a u,v --point-b u,v`。窗口缩放显示时程序自动还原原图坐标。按 Backspace 撤销，Esc 取消。输出包含 `distance_m`、两个地面坐标和标定残差。超出控制点包围区域默认拒绝；`--allow-extrapolation` 仅供专门核验，不能视作已验收测量。图片必须与标定原图字节一致，重编码或叠加新 ROI 后须重新标定。

## 悬空点三维距离

若用户点击的是工件两角、人员头顶与工件最低点等不同高度的点，需要同时取得至少两台相机的对应画面。在每路**同一帧**画面布置至少 6 个分布在三维空间的实测控制点和 2 个独立校验点，格式与地面点类似，但用 `xyz_m: [x,y,z]`。若控制点全在地面，不能通过当前 `calibrate-pose`；应增加不同高度且可见的标定靶，或另设计平面 PnP 流程。各相机的三维点须使用同一米制坐标系。

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance calibrate-pose --image 'E:\images\D02.png' --points 'E:\calibration\D02-3d-points.json' --intrinsics 'E:\calibration\cam-D02\intrinsics.json' --max-check-error-px 2 --out 'E:\calibration\D02-pose.json'
```

另一相机同样执行。`views.json` 格式：

```json
{
  "views": [
    {"camera_id": "D02", "image": "E:/images/D02.png", "pose": "E:/calibration/D02-pose.json", "timestamp_ms": 1000},
    {"camera_id": "D03", "image": "E:/images/D03.png", "pose": "E:/calibration/D03-pose.json", "timestamp_ms": 1001}
  ]
}
```

时间戳仅演示字段格式，必须替换为同一时间基准下的真实**曝光时刻**。画面烧录到秒的时间不能作为这里的毫秒同步证据。每路可填 `"pixels": [[A_u,A_v],[B_u,B_v]]`，省略则逐路弹窗点击 A、B；须确认各相机 A 指向同一个物理特征、B 也一样。纯轮廓角点在不同视角可能不是同一表面点，不可直接拿来三角化。

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance measure-3d --views 'E:\calibration\views.json' --max-time-offset-ms 5 --max-reprojection-error-px 2 --min-ray-angle-deg 1 --out 'E:\results\measure-3d.json'
```

上述阈值仅为命令语法示例，必须依据目标速度、允许位置误差、像素定位误差和现场验收制定。程序检查同步偏差、交会角、正深度、重投影误差，但**沿极线的错配也可能重投影得很好**；三维结果仍需同名点可信度和独立实测距离验证。物体最低点如果被遮挡，点击可见轮廓不能保证得到真正最低点。

## 验收与当前状态

1. 每台正式相机在最终安装、变焦、码流、镜头状态下采集棋盘格和控制点；运动位置变化时重新估计该帧位姿或建立经实测验证的自动跟踪机制。
2. 按近/中/远、画面中心/边缘、不同地面位置和不同相机车位布置**独立于拟合点**的已知距离；报告绝对误差分布、最大误差、不可判定率，不以拟合残差代替测距精度。
3. 现有四路视频、截图和设计说明缺少控制点实测坐标、实际内参及帧级同步证明，所以目前不能给出其中任意两点的可信米数；代码与合成数据测试通过，现场测量性能**尚未验证**。





配置文件是[camera_presets.json](/E:/横向项目-目标检测/pixel_distance/camera_presets.json)。运行时通过 `--camera D01` 至 `D04` 选择其中一组参数。

| 字段                                | 含义                                                         | 是否影响计算 |
| ----------------------------------- | ------------------------------------------------------------ | ------------ |
| `height_m`                          | 相机光心距地面的假定高度，单位米。当前取设计范围 8–12 米的中间值 **10 米**。 | **是**       |
| `horizontal_fov_deg`                | 当前镜头的假定水平视场角，单位度。角度越大，同样的像素间隔通常对应更大的实际范围。 | **是**       |
| `down_tilt_deg`                     | 镜头中心朝向相对水平线的向下俯角：`0°` 近水平，`90°` 垂直向下。当前斜视相机取 `45°`，近俯视相机取 `85°`，均为**未实测的初始假设**。 | **是**       |
| `roll_deg`                          | 相机绕镜头光轴的旋转角，用于描述画面倾斜；当前暂设 `0°`。    | **是**       |
| `design_height_range_m`             | 设计说明给出的高度范围 `[8,12]`。仅用于显示“只改变高度时距离会如何变化”，**不参与主结果**。 | 否           |
| `expected_aspect_ratio`             | 预期画面宽高比，当前为 `16:9`；用于拒绝明显不匹配的图片。    | 仅用于检查   |
| `name`、`mount`、`source`、`schema` | 相机名称、安装位置、参数来源说明和配置格式标记。             | 否           |

此外，命令行 `--plane-height-m` 指定**两个点共同所在水平面**的离地高度，默认 `0`，即两点都在地面。它不在相机配置文件中，因为它描述的是本次所选的点。

目前程序根据视场角推算焦距，并暂定主点在图像中心、像素为正方形、**忽略镜头畸变**。因此最值得优先替换为实测值的是 `height_m`、`horizontal_fov_deg` 和 `down_tilt_deg`；尤其在广角画面边缘，当前结果只能作为粗略估算。
