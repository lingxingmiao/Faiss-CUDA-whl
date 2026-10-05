# Faiss-CUDA-whl

自动构建 **覆盖 Maxwell 到 Hopper** 的 faiss GPU Windows 预编译包（wheel + 可直接替换的 `faiss.dll`；
Blackwell 等更新架构靠内嵌 PTX 由驱动 JIT），
上游 [`facebookresearch/faiss`](https://github.com/facebookresearch/faiss) 发一版，这里构建一版。

## 为什么需要它

conda-forge 与 PyPI 提供的 faiss-gpu Windows 二进制，`cuobjdump --list-elf` 显示内置 CUDA 架构恒为：

```
sm_53  sm_62  sm_72  sm_75  sm_80  sm_86  sm_89  sm_90      (PTX 仅 sm_90)
```

**没有 `sm_60`（Tesla P100）和 `sm_70`（Tesla V100）**，后果是：

| 构建 | 现象 |
| --- | --- |
| cuda130（CUDA 13） | `StandardGpuResourcesImpl::initializeForDevice` cublas 断言崩溃（CUDA 13 已放弃 Volta/Pascal） |
| cuda129 / cuda126 / cuda118（CUDA 12.x / 11.x） | `CUDA error 209 no kernel image is available for execution on the device` |

已核对 `faiss` 1.8.0 / 1.9.0 / 1.14.3 的 cuda118 / cuda120 / cuda126 / cuda129 / cuda130 全部变体，架构列表一致缺失。
因此只能在 CUDA 12.x 下**自行编译**并显式指定架构 —— 这正是本仓库做的事。

## 产物

每次构建（矩阵 = CUDA 版本 × Python 版本）产出：

| 产物 | 说明 |
| --- | --- |
| `faiss_gpu_<cuXXX>-<ver>+cuXXX>-cp3XX-cp3XX-win_amd64.whl` | 精简 wheel：内含 `faiss.dll` / `_swigfaiss.pyd`，`<发行版名>.libs/` 里由 delvewheel 放 `faiss.dll`、`openblas.dll`、`msvcp140.dll`、`vcomp140.dll`；**不含** CUDA 运行时（由系统 CUDA 或 `nvidia-*-cu12` pip 包提供） |
| `faiss_gpu_<cuXXX>-<ver>+cuXXX.full-cp3XX-cp3XX-win_amd64.whl` | （可选，`bundle_cuda`）内嵌 CUDA 运行时，约 1GB，完全离线可用。**此分支尚未在 CI 实跑验证过** |
| `faiss-<ver>-cuda<cuXXX>-cp3XX-win64-dll.zip` | conda / 已有 faiss 环境的落地包：`faiss.dll` + `openblas.dll` + README（DLL 与 Python 版本无关，文件名带 py 标签只为区分矩阵产物） |

轮子内的 `faiss/_gpu_build.py` 标记 + 补丁 0003 让 Windows 下自动
`os.add_dll_directory()` 到 `nvidia/{cuda_runtime,cublas,curand,nvjitlink}/bin` 与 `CUDA_PATH\bin`。

## 使用方法

### pip（推荐）

```powershell
# CUDA 12.9 / Python 3.12 为例，从 Release 下载后：
pip install faiss_gpu_cu129-1.15.1+cu129-cp312-cp312-win_amd64.whl

# 若本机没有 CUDA Toolkit，再装运行时 pip 包：
pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-curand-cu12

python -c "import faiss; print(faiss.__version__, faiss.get_num_gpus())"
```

### conda / 已装 faiss 的环境（落地 dll）

```powershell
:: 备份原 DLL 后覆盖（_swigfaiss.pyd 无需更换）
copy /y "%CONDA_PREFIX%\Library\bin\faiss.dll" "%CONDA_PREFIX%\Library\bin\faiss.dll.bak"
copy /y faiss.dll     "%CONDA_PREFIX%\Library\bin\faiss.dll"
copy /y openblas.dll  "%CONDA_PREFIX%\Library\bin\openblas.dll"
python -c "import faiss; print(faiss.__version__, faiss.get_num_gpus())"
```

> 自编译 DLL 链接 `openblas.dll`（conda 官方版链接的是 MKL 的 `libblas.dll` / `liblapack.dll`），
> 所以落地包必须带上 `openblas.dll`。回滚只需把 `faiss.dll.bak` 改回 `faiss.dll`。

## 自动构建机制

```
每天 03:17 UTC ─┐
手动 dispatch   ├─> plan  ──> (已发布过该版本? 跳过) ──> build 矩阵 ──> release
push main       │      查上游最新 release tag           windows-2022     汇总 assets
repository_dispatch                                    多 CUDA × 多 PY   tag: faiss-<ver>
```

* `plan`（[`scripts/plan_build.py`](scripts/plan_build.py)）：取上游最新 tag，检查本仓库是否已发布 `faiss-<tag>`；
  已发布则整个 workflow 直接结束（幂等，定时任务不会重复构建）。
* `build`（matrix）：Windows 上 `clone -> 打补丁 -> 计算架构 -> cmake+ninja -> bdist_wheel -> delvewheel -> 验证`。
* `release`：把全部产物挂到 `faiss-<ver>` 这个 Release 上。

手动触发可以覆盖：`version` / `cuda` / `python` / `arch_mode` / `bundle_cuda` / `strict_patches` / `force`。
新增 CUDA 版本只需在 `cuda` 输入里加（CUDA ≥ 13 会被自动拒绝，它无法编译 sm_50~sm_75）。
MSVC 不固定工具集、也不用在脚本里登记版本表：一律用 Runner 自带工具集 + nvcc 的 `-allow-unsupported-compiler`。

`arch_mode` 说明（[`scripts/cuda_archs.py`](scripts/cuda_archs.py) 用 `nvcc --list-gpu-arch` 动态枚举）：

| 值 | 生成 | 含义 |
| --- | --- | --- |
| `major`（默认） | 每个大版本一个 cubin，如 cu129 是 `50;60;70;80;90`（+`90-virtual`） | SASS 在同一大版本内小号向前兼容，所以 `sm_50` 覆盖 5.2/5.3、`sm_60` 覆盖 6.1/6.2、`sm_70` 覆盖 7.2/7.5 —— **一张 wheel 就能从 Maxwell 一路盖到 Hopper**。最低档由矩阵里的 `min_sm` 决定：cu118 = 35（GTX 780/Titan/K80），cu12x = 50（**GTX 745/750/900 系**起步） |
| `min60` | 全部 ≥ sm_60 的架构 | 从 Pascal 起，每档都出原生 cubin |
| `all` | nvcc 支持的一切 | cu12.9 上是 19 档（`sm_50`…`sm_121`）；小号架构本来靠 minor 向前兼容，一般用 `major` 就够 |
| 显式列表，如 `60,70` | `60-real;70-real;70-virtual` | 每个架构一趟 nvcc，构建时间与架构数成正比。只想跑 P100/V100 时用它（CI 上能省几倍时间），`--require` 仍会挡住写错的架构 |

> 构建时间基本等于「架构数 × 一遍完整 faiss CUDA 编译」（3 趟实测 24.8 分钟；7 趟按比例推算 ≈ 50 分钟）。
>
> **为什么默认封在 sm_90（`ARCH_MAX_SM = 90`）**：cu12.8/12.9 的 nvcc 能编 `sm_100/103/120`
> （Blackwell），但那批 CCCL 头文件（`cuda/__ptx/.../clusterlaunchcontrol.h`）在当前
> nvcc + MSVC 组合下会报 `asm operand type size(4) does not match type/size implied by
> constraint 'l'`。封在 90 后 Blackwell 显卡（RTX 50 系 / B200）仍能用，靠的是 `90-virtual`
> 的 PTX 由驱动 JIT；等 CCCL 修好把 `ARCH_MAX_SM` 改成 `0` 即可解封。
>
> **GTX 600 系（Kepler GK104 = sm_30）编不了**：`sm_30` 在 CUDA 11.0 就被移除，任何现役
> 工具链都出不了它的内核 → 这类卡用 CPU（TranslatorMinecraft 会自动回退，不会崩）。
>
> **CUDA 11.8（能编 sm_35：GTX 780/Ti、Titan、Tesla K80）走的是单独的 legacy 分支，尚未开放**：
> ① conda-forge 的拆分 CUDA 包（`cuda-nvcc`/`cuda-cudart-dev`/`cuda-profiler-api`/
> `libcublas-dev`…）最早只有 12.0，11.8 的组件得从 NVIDIA 的 redist 归档取；
> ② nvcc 11.8 不认 `-std=c++20`（实测 `nvcc fatal : Value 'c++20' is not defined for option 'std'`），
> 而 **faiss ≥ 1.14.3 声明 C++20**（1.13.2 / 1.14.0 还是 C++17）——所以这条线最高只能配
> **faiss 1.14.3**，并把 CUDA 侧压到 C++17（`-DCMAKE_CUDA_STANDARD=17`，本地已确认 1.14.3 的 CUDA
> 代码能在 C++17 下编译，四个补丁对 1.14.x 也全部适用）；
> ③ 还必须配 **MSVC 14.29(v142)**：更新的 STL 会直接 `static assertion failed: STL1002:
> Unexpected compiler version, expected CUDA 12.4 or newer` 把 nvcc 11.8 挡在门外。
> `plan_build.py` 现在遇到 CUDA < 12 默认跳过并打印这条理由（`allow_legacy_cuda` 开关落地后才放行）。

工具链来源（保证可复现，不依赖第三方 action 的版本表）：

- Python / OpenBLAS / CMake / Ninja / CUDA 全部由 **conda-forge** 提供，一次 `micromamba` 创建：
  `python=3.x numpy swig openblas cmake ninja cuda-version=12.9 cuda-nvcc cuda-cudart-dev cuda-profiler-api cuda-cuobjdump libcublas-dev libcurand-dev`
  （`cuda-profiler-api` 别漏：`faiss/gpu/utils/DeviceUtils.cu` 要 `cuda_profiler_api.h`，conda 的 CUDA 是拆包的，少这个包会在编译到一半时才报缺头文件），
  由 `cuda-version` 把各组件钉在同一条 12.x 线上。
- MSVC 不固定工具集（用 Runner 自带），改由 nvcc 的 `-allow-unsupported-compiler` 放开版本检查。
- **默认只出两条 CUDA 线：12.6 与 12.9**（`DEFAULT_CUDA`），Python 是 3.10~3.14，即 10 个 job；
  需要 12.4 / 12.8 时在 dispatch 的 `cuda` 输入里临时加（产物只是运行时对齐不同，见下）。
  CUDA 13 会被自动拒绝，11.x 也会被跳过（faiss ≥1.12 要 C++20，nvcc 11.8 不认）。

## 补丁集

[`.github/patches/`](.github/patches) 里的 4 个补丁由本仓库维护，`git apply` 应用；
若上游已修复（反向可应用）则自动跳过，`strict_patches=true` 时不可应用即失败。

| 补丁 | 内容 | 是否必需 |
| --- | --- | --- |
| `0001-msvc-rpcndr-small-macro.patch` | `faiss/gpu/utils/MergeNetworkWarp.cuh` 的 `bool small` 被 Windows SDK `rpcndr.h` 的 `#define small char` 破坏 → 重命名 `isSmall` | Windows 必需 |
| `0002-nvcc-msvc-view-initlist.patch` | `PQCodeDistances-inl.cuh` 中 nvcc 12.x 不接受 `view<2>({...})`（`expected an expression`）且 MSVC 需要 `template` 消歧 | Windows 必需 |
| `0003-windows-preload-cuda-dll.patch` | `faiss/python/__init__.py::_preload_gpu_libs` 增加 Windows 分支：注册 `nvidia-*-cu*` / `CUDA_PATH` 的 bin 目录（当前按 **1.15.x** 生成；上游重写该函数后需按新代码重新生成） | wheel 可用性 |
| `0004-wheel-ship-gpu-build-marker.patch` | `faiss/python/setup.py` 把 `_gpu_build.py` 打进 wheel，否则 GPU 预加载根本不会执行 | wheel 可用性 |

## 校验

每个 job 都会跑 [`scripts/verify_artifact.py`](scripts/verify_artifact.py)：

1. `cuobjdump --list-elf/--list-ptx` 断言自编译 `faiss.dll` **含 sm_60 与 sm_70**，且架构数 ≥ 2（只显式构建 `60,70` 时就是 2 档；默认矩阵是 `50,60,70,80,90`）；
2. wheel 内必须有 `faiss/_swigfaiss*.pyd` 与 `faiss/_gpu_build.py`；
3. 精简 wheel 不得内嵌 `cublas`；全打包 wheel 必须内嵌 CUDA 运行时；
4. 安装后 `import faiss` 冒烟测试；全打包 wheel 会先把 CUDA 从 `PATH` 移除再测。

GitHub 托管 runner 没有 GPU，所以 CI 只能验证「能加载 + 架构正确」；
真实 GPU 上的搜索结果一致性需在本地跑（本仓库维护者已在 V100/P100 上验证：
IndexShards/IndexReplicas 与 CPU 结果逐位一致，20w×768 单查询约 0.036 ms）。

## 目录结构

```
.github/
  workflows/build-faiss-cuda.yml   主工作流（plan / build / release）
  patches/*.patch                  4 个补丁
scripts/
  plan_build.py                    解析版本 + 生成矩阵 + 幂等跳过
  cuda_archs.py                    用 nvcc --list-gpu-arch 生成 CMake 架构串
  apply_patches.py                 打补丁（可识别“上游已修复”）
  verify_artifact.py               校验 DLL 架构 / wheel 内容
```

## 维护者笔记（踩过的坑，改工作流前先看）

Windows + micromamba + conda-forge CUDA 这套组合有几个反直觉的地方，都已在工作流里处理：

1. **构建步骤不做 micromamba 激活**：只用 `micromamba create` 建环境，之后把
   `<prefix>;<prefix>\Library\bin;<prefix>\Scripts` 塞到 PATH 前面直接调 cmake/ninja/python。
   原因有三：micromamba 1.x/2.x 的 `run` 参数不兼容（2.x 删了 `--no-capture-output`，传了会被
   当成要执行的命令）；Windows 激活脚本的 `SET` 回显会污染子进程 stdout；最关键的是**它会把
   conda 自带 MSVC(14.29) 的 INCLUDE/LIB 盖到 Runner 的 14.44 上**，导致 nvcc 自己去调
   `vcvars64.bat` 并在超长 PATH 上失败（`nvcc fatal : Could not set up the environment ...`）。
   这样改跟本地已验证的 `_faissbuild\configure.bat` 完全一致。
2. **CUDA 标准别乱钉，host 编译器要指全路径**：faiss 1.15 的 `CMAKE_CXX_STANDARD` 是 20，
   CMake 会把它传给 nvcc；钉成 17 会让 `approx_topk.h` 那批 C++20 头文件编不过。反过来，
   Runner 上同时有 VS2022(14.44) 和 VS2019(14.29)，只写 `cl.exe` 时 CMake 会挑到 14.29，
   nvcc 只会打印一行 `The -std=c++20 flag is not supported with the configured host compiler`
   然后静默退回 C++14。所以 `cl.exe` 要从 `$env:VCToolsInstallDir` 取全路径喂给
   `CMAKE_C_COMPILER/CMAKE_CXX_COMPILER/CMAKE_CUDA_HOST_COMPILER`，并加一个 5 秒的
   nvcc C++20 探针提前失败。
3. **补丁必须按 LF 应用**：Git for Windows 默认 `core.autocrlf=true`，`actions/checkout`
   会把 `.patch` 签出成 CRLF，而 faiss 源码在被 `core.autocrlf=false` 之后的 clone 里是 LF，
   于是每个上下文行都多一个 CR → `patch does not apply`（看起来像上游改了代码）。
   `.gitattributes` 固定 LF，`apply_patches.py` 应用前还会再做一次 CRLF→LF 归一化。
4. **`faiss/python/setup.py` 只认 `Release/` 子目录**（它是给 VS 多配置生成器写的），而
   Ninja 把 `_swigfaiss.pyd` 直接放在 `build/faiss/python/`。打包前需要把它拷进 `Release/`。
5. **wheel 要重新打标签**：setup.py 没有 `ext_modules`，`bdist_wheel` 会按纯 Python 打成
   `py3-none-any`，而里面是 ABI 相关的 `.pyd`。用 `wheel tags --python-tag cp3XX --abi-tag cp3XX
   --platform-tag win_amd64` 修正。
6. **delvewheel 的 `--add-path` / `--exclude` 是「分号分隔的单个参数」**，重复传只有最后一个
   生效（原来精简 wheel 因此既没排除 cublas，也找不到 `faiss.dll`）。`faiss.dll` 所在目录要
   显式加进 `--add-path`。
7. **冒烟测试的解释器是干净的 `actions/setup-python`**，`pip install --no-deps` 不会带 numpy，
   得自己装。
8. **upload-artifact 会按公共父目录保留结构**：`out/*.whl` + 仓库根的 `*.zip` 一起上传，
   artifact 里就变成 `out/xxx.whl` + `xxx.zip` 两级，release 的 `dist/*` 只匹配得到 zip、
   静默少挂 wheel。现在所有产物统一写进 `out/`，`BUILD-INFO` 也带 cuda/py 标签（矩阵里同名会互相覆盖）。
9. **`verify_artifact.py` 找 vendored 目录要按 `.libs/` 后缀**：delvewheel 是按发行版名建目录的
   （`faiss_gpu_cu129.libs`），写死 `faiss.libs` 会让「精简 wheel 不得内嵌 cublas」这条检查
   实际上什么都没看。

## 许可

补丁仅针对 faiss 构建，faiss 本体为 MIT License，版权归 Meta Platforms, Inc. 所有。
