# Rebocap Lite

运行依赖：`numpy`、`mujoco`、`mink`。视频录制额外需要 `imageio[ffmpeg]`。

```bash
python /home/xhz/GMR/rebocap/rebocap_live_to_robot_lite.py \
  --robot semi_taks_lv1 --port 9010 --motion_fps 30 \
  --save_path /home/xhz/GMR/data/Mixamo/rebocap_live_semi_taks_lv1.pkl \
  --rate_limit
```

无图形界面运行时增加 `--no-viewer`。参数 `--rotation-mode`、`--no-rest-offset`、`--record_video` 与原程序一致。

脚本自带 `template.txt`、LV1 MuJoCo 资源和 IK 配置，不依赖 `libs/drivers/rebocap` 下的接收器、BVH 转换器或 GMR 的 PyTorch 保存模块。
