# 当前权重验证记录

验证日期：2026-10-19（Asia/Kuala_Lumpur）。

**当前状态：验收通过。** 当前 HVI-CIDNet 使用用户放在项目父目录的 `model.safetensors` 新权重（配置为 `../model.safetensors`）。两个方法均完成 LOLv1 全部 15 对测试图评测，运行状态为 `completed`，退出码为 0。不使用 GT 均值校正。

## 当前真实结果

运行编号：`20261019T083852.418361Z`。

| 方法 | PSNR ↑ | SSIM ↑ | LPIPS ↓ | 平均单张推理 | 验收 |
| --- | ---: | ---: | ---: | ---: | --- |
| Zero-DCE | 14.8611 dB | 0.5591 | 0.3349 | 0.0361 s | 通过 |
| HVI-CIDNet（用户提供权重） | 23.6153 dB | 0.8710 | 0.1061 | 1.3858 s | 通过 |

全流程墙钟时间约 **26.12 秒**，包含模型及指标初始化、预热、推理、指标和读写；不包含依赖安装及下载。Zero-DCE 最慢单张 0.0405 秒，HVI-CIDNet 最慢单张 1.4415 秒，均满足配置中的耗时要求。

HVI-CIDNet 使用用户于 2026-10-19 明确确认的验收区间：PSNR [23,25] dB、SSIM [0.85,0.92]，单张推理时间不超过 60 秒。Zero-DCE 保留原任务包判据。

## 权重格式及当前配置

`config.yaml` 的 HVI 权重路径为 `../model.safetensors`。适配器对 `.safetensors` 明确执行：

```python
from safetensors.torch import load_file
state_dict = load_file(str(weight_path), device="cpu")
model.load_state_dict(state_dict, strict=True)
```

只有 `.pth` 等 PyTorch checkpoint 格式才使用 PyTorch 权重读取方式。当前文件的 191 个参数张量均已严格加载，没有跳过缺失参数或随机初始化。

当前文件 SHA256：`928bc914bcec03abd6f2b9e057007961845a75e156b0536ba7c41bb4ac817c4c`。

这份文件与此前下载的官方 wperc 权重不同，191 个参数中有 178 个张量不同，因此不能沿用此前的 23.8436 dB 结果。当前文件的具体 Hugging Face 模型版本未提供，故在记录中标注为“用户提供权重”，不推断其训练版本。

此前 wperc 的权重、数值与评测口径排查完整保留在 [历史验证记录](validation-wperc.md) 及 `examples/lolv1-wperc/`，其中包含与官方结果对齐的 8 位截断对照。当前样例目录 `examples/lolv1/` 则只对应当前配置。

## 已完成的检查

- 当前数据集 `LOLdataset/eval15` 的 30 个图像文件已与作者原始数据包逐文件核对 SHA256，一致；未使用 `our485` 训练集。
- 当前两个方法的全部 30 份 float32 输出均重新计算 PSNR/SSIM，与对应 CSV 行一致。CSV 包含 30 个逐张行及 2 个均值行，均标记为 `pretrained`。
- 输入与输出统一为 RGB、float32、[0,1]、原尺寸；不裁边、不做 GT 均值校正。SSIM 使用 11×11 高斯窗口，LPIPS 使用预训练 AlexNet。
- 新 HVI 权重已额外核对 CPU 单图重复推理及 CPU/MPS 一致性；结果见 `examples/lolv1/device_verification.json`。Zero-DCE 的该项记录沿用相同权重的先前检查。
- 配对、范围、指标、缺失权重等 7 项核心测试已通过；随机初始化流程另行验证并标记为 `completed_unvalidated`，未混入正式结果。
- 模型代码和统一指标实现保持原有推理口径；本次仅明确 safetensors 读取方式并切换用户提供的权重。官方网络源码未修改。
- 当前 CSV、汇总表、对比拼图及 `run.json` 已更新，记录了当前权重 SHA256 和验收状态。

环境仍为 Python 3.11.17、PyTorch 2.14.1、macOS arm64；增强使用 MPS，LPIPS 使用 CPU。配置、完整依赖版本和固定官方源码提交分别见 `config.yaml`、`requirements-lock-macos-arm64.txt` 和 `sources.json`。

当前本机原始运行记录在 `outputs/20261019T083852.418361Z/`。可随仓库展示的样例在 `examples/lolv1/`，其运行元数据已移除本机绝对路径。
