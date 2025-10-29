## LivePortrait 全链路 GPU 环境改造与问题排查备忘

更新时间：2025-08-10

---

### 1) 硬件与系统
- **OS**: Ubuntu 24.04 LTS
- **GPU**: NVIDIA GeForce RTX 5070（Blackwell，sm_120）
- **Driver / CUDA runtime**: `nvidia-smi` 显示 CUDA 12.8
- **nvcc (Toolkit)**: 12.0（仅用于本地编译；不影响 PyTorch/ORT 运行时）

---

### 2) Conda 环境
```bash
conda create -n LivePortrait python=3.10 -y
conda activate LivePortrait
```

> 注意：后续所有 pip 操作均在 `LivePortrait` 环境内执行，统一使用 `python -m pip` 保证解释器一致。

---

### 3) PyTorch（支持 RTX 5070 / sm_120）
RTX 5070 需较新 PyTorch（支持 sm_120）。选择 CUDA 12.8 轮子：
```bash
python -m pip install --no-cache-dir \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
```
验证：
```bash
python - << 'PY'
import torch
print('torch=', torch.__version__, 'cuda=', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
PY
```

---

### 4) OpenCV 与 NumPy/SciPy 兼容
- 避免 OpenCV 4.10.0 的 `resize` 异常；固定为 4.9.0.80（headless）。
- 避免 NumPy 2.x 的 ABI 兼容问题；固定 NumPy/SciPy 到 1.x/1.11。

```bash
python -m pip uninstall -y opencv-python opencv-python-headless numpy
python -m pip install --no-cache-dir "numpy==1.26.4" "scipy==1.11.4"
python -m pip install --no-cache-dir --no-deps "opencv-python-headless==4.9.0.80"

# 验证
python - << 'PY'
import numpy, scipy, cv2
print('NumPy=', numpy.__version__, 'SciPy=', scipy.__version__, 'OpenCV=', cv2.__version__)
PY

# 基本功能测试
python - << 'PY'
import numpy as np, cv2
img = np.random.randint(0,255,(100,100,3), dtype=np.uint8)
r = cv2.resize(img,(50,50))
print('resize ok:', r.shape)
PY
```

---

### 5) ONNXRuntime 对齐 CUDA 12（启用 CUDA/TensorRT Provider）
安装与 CUDA 12 期匹配的 onnxruntime-gpu，并指向 pip 安装的 NVIDIA 动态库：
```bash
python -m pip uninstall -y onnxruntime onnxruntime-gpu
python -m pip install --no-cache-dir onnxruntime-gpu==1.20.1
```
可选安装 cuDNN v9（供 ORT 使用）：
```bash
python -m pip install --no-cache-dir "nvidia-cudnn-cu12>=9.6,<10"
```
临时加入库路径（当前 Shell 验证）：
```bash
PY_SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:\
$PY_SITE/nvidia/cublas/lib:\
$PY_SITE/nvidia/cuda_runtime/lib:\
$PY_SITE/nvidia/cudnn/lib:\
$PY_SITE/nvidia/cufft/lib:\
$PY_SITE/nvidia/curand/lib:\
$PY_SITE/nvidia/cusparse/lib:\
$PY_SITE/nvidia/cusolver/lib:\
$PY_SITE/nvidia/nccl/lib:\
$PY_SITE/nvidia/nvtx/lib:\
$PY_SITE/nvidia/nvjitlink/lib
```
验证 Provider 与依赖解析：
```bash
python - << 'PY'
import onnxruntime as ort
print('ort=', ort.__version__, 'providers=', ort.get_available_providers())
PY

# 确认依赖 .so 已找到（应为 libcublasLt.so.12 与 libcudnn*.so.9）
ldd $(python - << 'PY'
import onnxruntime, pathlib
print(pathlib.Path(onnxruntime.__file__).parent/'capi'/'libonnxruntime_providers_cuda.so')
PY) | grep -E 'cublasLt|cudnn'
```

> 不建议用软链把 `.12` 伪装成 `.11`，ABI 不兼容风险高。若仍出现 `.11 not found`，请更换 ORT 版本（如 1.19.2/更高），直至依赖变为 `.12`。

---

### 6) 让库路径随环境激活自动生效（推荐）
使用 Conda 环境钩子，仅在 `LivePortrait` 环境中追加：
```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/onnxruntime_cuda.sh" <<'EOF'
# LivePortrait: add pip-installed CUDA 12 libs
PY_SITE="$(python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)"
export OLD_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"
if [ -n "$PY_SITE" ] && [ -d "$PY_SITE/nvidia" ]; then
  export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$PY_SITE/nvidia/cublas/lib:$PY_SITE/nvidia/cuda_runtime/lib:$PY_SITE/nvidia/cudnn/lib:$PY_SITE/nvidia/cufft/lib:$PY_SITE/nvidia/curand/lib:$PY_SITE/nvidia/cusparse/lib:$PY_SITE/nvidia/cusolver/lib:$PY_SITE/nvidia/nccl/lib:$PY_SITE/nvidia/nvtx/lib:$PY_SITE/nvidia/nvjitlink/lib"
fi
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/onnxruntime_cuda.sh" <<'EOF'
# LivePortrait: restore LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$OLD_LD_LIBRARY_PATH"
unset OLD_LD_LIBRARY_PATH
EOF
```

> 如你使用 `~/.bashrc` 持久化，请用 `if [ "$CONDA_DEFAULT_ENV" = "LivePortrait" ] ; then ... fi` 包裹，避免污染 base 环境。

---

### 7) 运行与常用参数
- 快速体验（人类模型）
```bash
python inference.py
# 或指定素材
python inference.py -s assets/examples/source/s13.mp4 -d assets/examples/driving/d0.mp4
# 复用已缓存驱动模板（更快）：
python inference.py -s assets/examples/source/s13.mp4 -d assets/examples/driving/d0.pkl
```
- 输出：
  - `animations/<src>--<drv>.mp4`
  - `animations/<src>--<drv>_concat.mp4`
  - 首次运行会生成 `assets/examples/driving/d0.pkl`，下次可直接使用以跳过裁剪/模板构建。

---

### 8) 本次代码稳健性修改（最小化影响）
> 所有修改仅为兼容 PyTorch 2.7.x / NumPy 1.x/2.x 的数据类型与内存布局差异，避免 from_numpy/DLPack/Object 数组导致的错误；功能逻辑不变。

- `src/live_portrait_wrapper.py`
  - `prepare_source` / `prepare_videos`：将 `torch.from_numpy(...)` 改为 `torch.tensor(..., dtype=torch.float32)`，并保留 `permute` 到标准 NCHW；避免因非标准 ndarray（或 object dtype）触发 TypeError。

- `src/utils/io.py`
  - `dump(...)`：
    - `pickle.dump(..., protocol=pickle.HIGHEST_PROTOCOL)`；
    - 对嵌套结构中带 `__array__` 的对象统一 `np.asarray(...)`，避免 “Can't pickle <class 'numpy.ndarray'>: it's not the same object as numpy.ndarray”。

- `src/utils/helper.py`
  - `dct2device(...)`：
    - 对 numpy object 数组做判断与兜底转换（`np.asarray(..., dtype=np.float32)` / `.tolist()` 路径），失败则跳过该键，避免 DLPack 与 object dtype 报错。

- `src/live_portrait_pipeline.py`
  - 在取出 `R_d_i`/缓存 `x_d_0_info` 时，统一转换为 `torch.Tensor(float32, device)`；
  - 对 object 数组补充 `tolist()` -> Tensor 的兜底路径；
  - 避免 `numpy.ndarray` 调用 `permute` 与 “can't convert cuda tensor to numpy” 等错误。

> 备注：若后续将 NumPy 升级到 2.x，以上转换逻辑仍能规避大多数不兼容问题。

---

### 9) 常见报错与处理速查
- `OpenCV(4.10.0) resize: Overload resolution failed` → 降级到 `opencv-python(-headless)==4.9.0.80`。
- `TypeError: expected np.ndarray (got numpy.ndarray)` / `DLPack only supports ...` → 使用 `torch.tensor(...)` 代替 `from_numpy`，并对 object 数组 `.tolist()` 兜底。
- `PicklingError: Can't pickle <class 'numpy.ndarray'>` → `pickle.HIGHEST_PROTOCOL` + 递归 `np.asarray` 标准化。
- `AttributeError: 'numpy.ndarray' object has no attribute 'permute'` → 确保矩阵变量为 `torch.Tensor` 再做 `permute`。
- `libcublasLt.so.11 not found` → 安装 CUDA 12 构建的 ORT（如 `onnxruntime-gpu==1.20.1`），并加入 pip nvidia 动态库路径。
- `libcudnn_adv.so.9 not found` → `pip install nvidia-cudnn-cu12` + 追加 `$PY_SITE/nvidia/cudnn/lib` 到 `LD_LIBRARY_PATH`。
- `CPU dispatcher tracer already initialized` + `numpy._core.multiarray failed to import` → 固定 `numpy==1.26.4` + `scipy==1.11.4`，避免 ABI 冲突。

---

### 10) 一键自检脚本（可选）
```bash
python - << 'PY'
import torch, onnxruntime as ort, cv2, numpy as np
print('torch=', torch.__version__, 'cuda=', torch.version.cuda, 'gpu=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('ort=', ort.__version__, 'providers=', ort.get_available_providers())
print('cv2=', cv2.__version__, 'numpy=', np.__version__)
PY
```

---

### 11) 注意事项
- 不要再执行 `pip install -r requirements.txt`（会把 ORT 降回 1.18.0，破坏 CUDA 12 对齐）。如需补包，请精准安装单个依赖。
- 若需持久使用系统 CUDA（如 `/usr/local/cuda-12.8`），可在环境钩子或 `~/.bashrc` 中追加 `export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH"`（建议同样仅在 `LivePortrait` 环境内生效）。

---

以上步骤已在本机验证：生成 `animations/<...>.mp4` 与 `animations/<...>_concat.mp4` 成功；ONNXRuntime 与 PyTorch 均可使用 GPU。
