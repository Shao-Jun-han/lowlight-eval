# LowLight-Eval

低光图像增强统一评估工具箱。每个方法只写一个 adapter，数据读取、指标、计时、结果表和拼图共用同一套流程。目前接入 **Zero-DCE** 与 **HVI-CIDNet**，使用官方模型和预训练权重，只做推理。

2026-10-19 经用户确认，HVI-CIDNet 的 PSNR 检查区间修订为 23–25 dB，继续不使用 GT 均值校正。当前配置使用用户放在项目父目录的 `model.safetensors`；该文件与先前下载的 wperc 权重不同，已用新权重完成全部 15 对测试图复测，两个方法均通过验收；具体结果见下方验证记录。

**English abstract.** LowLight-Eval is a small, extensible inference benchmark for low-light image enhancement. Two-function adapters integrate official Zero-DCE and HVI-CIDNet implementations with a shared LOLv1 evaluation pipeline, RGB PSNR/SSIM, pretrained AlexNet LPIPS, synchronized inference timing, and comparison figures. Apple Silicon MPS is preferred, with CPU fallback. No training or ground-truth brightness correction is used.

本机实测、验收状态和已知口径差异见 [验证记录](docs/validation.md)。[示例结果](examples/lolv1/results.csv) 包含真实 15 对测试图的逐张指标与均值，不使用随机模型或合成数据充数。

![LOLv1 对比图](examples/lolv1/comparison.png)

## 1. 安装与运行环境

建议 Python **3.11**。本机已经创建 `.venv`，进入目录并激活即可：

```bash
cd lowlight-eval
source .venv/bin/activate
python check_environment.py
```

换到一台新机器时，在项目目录创建环境并安装依赖：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/prepare_models.py
```

若使用 uv，也可以用 `uv venv --python 3.11 .venv` 创建环境。`requirements-lock-macos-arm64.txt` 记录了本次实测的完整依赖版本；要复用本机组合，可改为安装这份锁定文件。首次计算 LPIPS 会从 torchvision 官方地址下载约 233 MiB 的 AlexNet 权重，缓存在项目 `.cache/torch` 内。

`device: auto` 会优先选择 MPS，否则使用 CPU。需要 CPU 时，修改 `config.yaml` 为 `device: cpu`。如果受限执行环境内显示 MPS 不可用，请在本机终端复查；本次普通本机进程能正常使用 MPS。不需要 CUDA 或 NVIDIA 显卡。

## 2. 准备 LOLv1 数据

数据集来自 [作者项目页面](https://daooshee.github.io/BMVC2018website/)。完整 LOLv1 包含训练集 `our485` 和测试集 `eval15`，本项目只读取后者。

当前用户提供的数据放在项目旁边：

```text
Project/
├── LOLdataset/
│   ├── our485/          # 训练集，本次不读取
│   └── eval15/
│       ├── low/         # 15 张低光图
│       └── high/        # 同名的 15 张正常光图
└── lowlight-eval/
    └── config.yaml
```

修改 `config.yaml` 中的 `dataset.low_dir`、`dataset.high_dir` 即可使用其他位置。相对路径始终相对于配置文件所在目录，也支持绝对路径和 `~`。

```bash
python check_environment.py --check-data
python metrics.py --lpips
python -m unittest discover -s tests -v
```

数据检查必须列出 15 对图像。配对按文件名主干进行，不按两个列表的位置猜测。漏图、重名、尺寸不一致或误传入 485 对训练图都会报错。

## 3. 模型与权重

```bash
python scripts/prepare_models.py
```

脚本读取 `sources.json`，下载固定提交的官方仓库和作者公开的 HVI-CIDNet LOLv1 `wperc` 权重。Zero-DCE 使用官方仓库自带的 `Zero-DCE_code/snapshots/Epoch99.pth`。脚本下载的 HVI-CIDNet 权重保存在 `weights/hvi_cidnet_lolv1_wperc.safetensors`，并检查 SHA256。

**当前实际使用的是用户提供的 `../model.safetensors`**，由 `config.yaml` 的 `weight_path` 指定。HVI adapter 对 `.safetensors` 直接调用 `safetensors.torch.load_file(..., device="cpu")`，再严格加载模型参数；`.pth` 才使用 PyTorch 权重读取方式。换机器时请携带当前文件并配置相应路径；如要使用脚本下载的 wperc 权重，将配置改为 `weights/hvi_cidnet_lolv1_wperc.safetensors`。不同权重的结果分别记录，之前 wperc 的[验证记录](docs/validation-wperc.md)和样例保存在 `examples/lolv1-wperc/`。

官方网络源码放在 `vendor/`，不复制到 adapter，也不修改官方源码。适配器处理设备、权重格式、张量转换和尺寸填充。HVI-CIDNet 采用官方 `eval.py --lol` 的设置：`trans.gated=True`，默认 `alpha_s=1.3`，不做 GT 均值校正。

## 4. 按顺序运行

先确认单张结果：

```bash
python infer.py --method zero_dce --input ../LOLdataset/eval15/low/1.png --output outputs/zero_dce_single.png
```

再执行完整测试集：

```bash
python evaluate.py --methods zero_dce
python evaluate.py
```

第一条只评测 Zero-DCE。第二条按配置顺序先 Zero-DCE、后 HVI-CIDNet；每个方法的 sanity check 通过后才执行下一个。每次运行创建独立的 UTC 时间戳子目录，避免旧结果混入：

```text
outputs/<run_id>/
├── results.csv          # 逐张结果 + 每个方法的 __mean__ 行
├── summary.csv          # 每个方法一行均值
├── summary.md           # 可直接复制到 README 的表格
├── run.json             # 配置快照、版本、设备、权重/数据指纹、验收状态
└── images/<method>/
    ├── <image_id>.png   # 8 位展示图
    └── <image_id>.npy   # 实际被评测的 float32 数组
```

`results.csv` 的 `row_type` 区分单图与均值；再次聚合时只选择 `row_type=image`，避免把均值行重复算进去。`weights` 列区分 `pretrained` 与 `random_init`。

指标或耗时不满足配置时，程序保存已有结果，记录 `sanity_failed`，以退出码 **2** 结束。先检查 RGB/BGR、范围、测试划分，再核对权重、量化和文献口径。不能因数字偏离预期而自动改亮度或放宽判据。

生成拼图时，把终端输出的真实运行目录传给 `--run`：

```bash
python visualize.py --run outputs/<run_id> --images 1 22 780
```

拼图每行一个样本，每列依次为输入、各已完成方法和 GT，并显示单图指标与运行状态。可用 `--panel-width 600` 提高清晰度，或用 `--output` 指定位置。拼图的 PNG 展示精度与默认 float32 评测之间有轻微量化差别，原始数组已另存。

## 5. 固定评测口径

- `batch_size=1`；模型 `.eval()`、关闭参数梯度，推理和指标计算使用 `torch.no_grad()`。
- 输入输出统一为 RGB、H×W×3、float32、[0,1]；不缩放，不裁边，不使用 GT 亮度校正。
- PSNR：RGB 全图 MSE，`data_range=1`；完全相同图像返回 `inf`。
- SSIM：skimage 实现，11×11 高斯窗口，sigma=1.5，使用总体协方差，三个 RGB 通道取均值。
- LPIPS：`lpips==0.1.4`，默认预训练 AlexNet，输入映射为 [-1,1]，默认在 CPU 计算以减少 MPS 占用。
- 默认直接计算 float32 输出。`metrics.quantize_8bit: true` 可统一切换为先四舍五入量化再评测；比较时必须同时报告此设置。官方保存图像后的计算口径可能略有差别。
- 固定 Python / NumPy / PyTorch 种子为 42，固定 CPU 线程数。启用确定性检查；不支持的算子会警告。跨设备和库版本仍可能有微小数值差异，运行记录保留依赖版本与文件指纹。
- 每个模型先预热一次。计时包含 `enhance` 的设备传输、填充、推理和输出转回 CPU，前后同步 MPS；不包含模型加载、指标、读写文件。CSV 是逐张推理时间，`run.json` 另存整次运行的墙钟时间。

HVI-CIDNet 官方结果表区分是否使用 GT 平均亮度校正；高分结果不能直接用来验收不使用 GT 的输出。具体修订状态见验证记录。

## 6. 权重暂缺时的流程测试

缺失或损坏的权重会报错，绝不自动退回随机模型。需要明确测试随机初始化时：

```bash
python scripts/make_smoke_data.py --random-hvi
python evaluate.py --config artifacts/smoke_random.yaml --methods hvi_cidnet --allow-random
```

这会生成命名为 `SYNTHETIC_SMOKE_TEST` 的合成数据集，并让 HVI-CIDNet 的 `weight_path` 显式为 `null`。必须同时提供 `--allow-random` 才能运行。结果标记为 `random_init`、`completed_unvalidated`，不能算作复现通过。使用已有预训练权重测试整条流水线时，去掉 `--random-hvi` 并使用 `artifacts/smoke.yaml`。

## 7. 如何新增一个方法

在 `methods/new_method.py` 写两个函数：

```python
def load_model(weight_path, device):
    """读取官方结构、严格加载权重，返回在指定设备上的 eval 模型。"""
    ...

def enhance(model, img):
    """RGB numpy float32 [0,1] 输入，返回同尺寸、同格式结果。"""
    ...
```

在 `config.yaml` 的 `methods` 列表追加：

```yaml
- name: new_method
  module: methods.new_method
  enabled: true
  repo_path: vendor/official_new_method
  weight_path: weights/new_method.pth
```

`repo_path` 是该官方结构所需的导入根目录。可选的 `sanity` 包含 PSNR/SSIM 区间及单图时间上限；没有参考区间时会标记 `no_reference`，不会声称验收通过。统一流水线只通过 `load_model` 和 `enhance` 使用模型，不需要修改 `evaluate.py` 或 `visualize.py`。官方代码应独立放在 `vendor/`；若不同仓库的顶层模块重名，请在 adapter 内隔离导入。

## 文件导航

| 文件 | 用途 |
| --- | --- |
| `config.yaml` | 数据、设备、方法、权重、指标和验收配置 |
| `dataset.py` | 严格配对、RGB 读取、格式校验、PNG 保存 |
| `runtime.py` | 配置路径解析、种子、设备、adapter 加载 |
| `metrics.py` | 统一 PSNR / SSIM / LPIPS 与指标自测 |
| `methods/zero_dce.py`、`methods/hvi_cidnet.py` | 两个官方模型的适配器 |
| `infer.py`、`evaluate.py`、`visualize.py` | 单图、全量评测、对比拼图 |
| `scripts/prepare_models.py`、`sources.json` | 固定版本代码与权重获取 |
| `scripts/make_smoke_data.py`、`tests/test_core.py` | 合成流程测试与关键错误检查 |
| `docs/validation.md`、`examples/lolv1/` | 实测证据与可展示交付物 |

`vendor/`、`weights/`、`.venv/`、`__pycache__/`、`.cache/`、`outputs/`、`artifacts/` 和模型权重文件已忽略，提交 GitHub 时只包含代码、文档和精选示例。`outputs/` 在本地只保留最新运行 `20261019T083852.418361Z`；README 拼图使用仓库内的相对路径 `examples/lolv1/comparison.png`。

## 参考与使用范围

- [Zero-DCE 官方实现](https://github.com/Li-Chongyi/Zero-DCE)，CVPR 2020。其官方 README 将代码限定为学术研究用途并标注 CC BY-NC 4.0；本项目不改变该许可。
- [HVI-CIDNet 官方实现](https://github.com/Fediory/HVI-CIDNet)，CVPR 2025；[作者公开 LOLv1 权重](https://huggingface.co/Fediory/HVI-CIDNet-LOLv1-wperc)。仓库与权重各自保留上游声明。
- [LOLv1 作者页面](https://daooshee.github.io/BMVC2018website/)，BMVC 2018。
- [LPIPS 官方实现](https://github.com/richzhang/PerceptualSimilarity)。

本期必需方法已实现；CLAHE、直方图均衡和 Retinexformer 属于后续可选扩展。本项目不包含训练和对抗扰动实验。
