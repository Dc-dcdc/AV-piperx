# none / A→V Router 使用顺序

## 1. 采集反事实监督数据

```bash
/home/dc/miniforge3/envs/AV-piper/bin/python \
  train/s2_incremental/router/collect_none_av_router_dataset.py \
  router_collection.device=cuda:0 \
  router_collection.start_seed=2000 \
  router_collection.n_episodes=200
```

每个采样点从同一环境快照执行 `none` 和 `A→V` 两条8步候选，
随后两边都用冻结 `none` 策略续跑。默认每个episode最多采6个点，因此正式
采集会明显慢于普通评估。

可以更换不重叠seed区间多次运行。它们保存在同一模型/任务哈希目录的不同
`seed=<start>_ep=<count>` 子目录中，不会相互覆盖。

## 2. 联合训练Router

将上一步生成的manifest传给训练入口：

```bash
/home/dc/miniforge3/envs/AV-piper/bin/python \
  train/s2_incremental/train/train_none_av_router.py \
  'router_cache.manifests=[outputs/buffer/output_corrector_router/none_av_router_<hash>/seed=2000_ep=200/manifest.json,outputs/buffer/output_corrector_router/none_av_router_<hash>/seed=2200_ep=200/manifest.json]' \
  device=cuda:0 \
  wandb.enable=true
```

训练集和验证集按完整episode seed切分。视觉编码器、两个UNet和输出修正器
全部冻结，只更新约10.9万Router参数。每次验证会校准一个偏保守的概率阈值，
并将阈值写入完整复合策略checkpoint。

## 3. 同种子三组评估

在 `train/s2_incremental/eval/eval_none_av_router.py` 中填写Router checkpoint，
然后运行：

```bash
/home/dc/miniforge3/envs/AV-piper/bin/python \
  train/s2_incremental/eval/eval_none_av_router.py
```

脚本依次评估：

- `none`：精确冻结双扩散基线；
- `always_arm_to_view`：始终开启A→V；
- `learned_router`：使用验证集校准阈值。

三组共用相同episode seeds。结果额外记录Router决策次数、开启次数、开启率
和平均概率。
