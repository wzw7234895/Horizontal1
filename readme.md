# 车载边缘计算盒目标检测与像素距离计算

人员检测训练归档位于 [`person_tracking`](person_tracking/README.md)，包含 YOLO11n 训练代码、v13 人员标注数据集、训练过程文件、最优权重及训练记录。训练归档说明见 [`person_tracking/训练归档_v13_2026-11-08.md`](person_tracking/训练归档_v13_2026-11-08.md)。

`pixel_distance` 是当前项目的距离计算工具，支持：

- 地面控制点标定后的单相机地面距离；
- 已标定相机位姿下的多视角三维交会；
- 四路视频按时间抽帧、自动跨视角匹配和高度差候选估计；
- 可选的地面候选分割诊断。

## 运行环境

建议使用项目已有环境：

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m pip install -r pixel_distance\requirements.txt
```

运行帮助：

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance --help
```

详细的参数、标定前提、运行示例和现场限制见：

- [`pixel_distance/README.md`](pixel_distance/README.md)
- [`pixel_distance/当前测距方法与实现思路.md`](pixel_distance/当前测距方法与实现思路.md)
- [`pixel_distance/跨视角匹配现场复核_2026-11-07.md`](pixel_distance/跨视角匹配现场复核_2026-11-07.md)
- [`pixel_distance/运输车车长尺度核对_2026-11-07.md`](pixel_distance/运输车车长尺度核对_2026-11-07.md)

## 直接从四路视频测距

在参考画面中指定 A、B 两点，程序自动抽取同一时间的其他视角：

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m pixel_distance video-distance `
  --videos-dir 'E:\现场视频目录' `
  --time '2026-10-20T15:57:32' --reference D04 `
  --point-a 1278.17,445.71 --point-b 1254.86,922.97 `
  --out 'pixel_distance\results\measure.json'
```

输出会区分多视角三维候选、单目假定平面估算和无法确定。未完成现场内参、畸变、相对位姿及同步验收前，不应把名义估算作为真实测量值。

## 测试

```powershell
& 'D:\Anaconda\envs\crane-person\python.exe' -m pytest pixel_distance\tests -q
```
