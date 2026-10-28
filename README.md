# rlcode —— 上饶麻将强化学习（Suphx 特征 + PPO2）

> 江西上饶麻将规则的强化学习智能体。Suphx 式牌面特征编码 + stable-baselines PPO2 + 自定义 ResNet 策略网络，
> 用「智一」专家模型做对手与示范。工程内部名为 `shangrao_mj_rl_v4_suphx_cppy`。

## 项目简介

这是一个用自我博弈/专家对抗训练上饶麻将 AI 的研究型工程。整体分四层：

- **规则与对局层** `mahjong.py`：`Game2` → `Mcts_game` → `Game_RL` → `MahjongEnv` 四层继承，实现发牌、定精（宝牌）、吃碰杠胡判定、番型计分与单步推进；
- **状态层** `mah_state.py`：`RL_state` 把某一时刻某个座位的可见视野打包（手牌、副露、弃牌、剩余牌墙、精牌、赢牌结果等）；
- **接口层** `mahEnv.py`：把对局包装成 `gym.Env`，输出 Suphx 风格张量 + 合法动作掩码；
- **学习层** `sbln_learning/`：PPO2 训练脚本、自定义 ResNet 策略网络（TF1）、以及一套并行的 PyTorch PPO/DQN 实现。

「智一」不是本地模型，而是作者部署在远端 HTTP 服务上的专家推荐接口（`mah_tool/url_recommend.py`），用于生成示范轨迹和充当对手。

## 方法 / 模型

### 环境接口

`mahEnv.MahEnv(gym.Env)`：

| 项 | 值 |
| --- | --- |
| 动作空间 | `spaces.Discrete(34)` |
| 观测空间 | `spaces.Box(low=0, high=1, dtype=np.int, shape=(1274, 34, 1))`（声明值） |
| `reset()` | 返回 `(state, mask)` |
| `step(action)` | 返回 `(state, reward, done, info)`，`info["mask"]` 为 34 维合法动作掩码 |

34 个动作对应 `MahjongEnv.action_set = [1..9, 11..19, 21..29, 31..37]`，即万/条/筒各 9 张 + 7 张字牌。

观测是 **Suphx 风格的 one-hot 平面堆叠**：`mah_tool/tool2.py::get_suphx_1330_and_mask()` 先由 `card_preprocess_sr_suphx_1330()` 把状态编码成 `C × 34` 的多通道张量（通道数即特征维，基准 1330），再单独生成 34 维掩码：

```python
mask = [0] * 34
handCards = [translate3(f10_to_16(i)) for i in set(state.handcards)]
for item in handCards:
    mask[item] = 1          # 只在手牌上为 1
```

特征提取本体在 `mah_tool/suphx_extract_features/feature_extract.py`，按胡牌牌型分别计算向听/进张：`wait_types_comm_king`（平胡，带精牌）、`wait_types_7`（七对）、`wait_types_haohua7`（豪华七对）、`wait_types_13`（十三烂）、`wait_types_19`，再由 `suphx_cards_feature_code` / `suphx_data_feature_code` 拼成多通道。

> 注意：`observation_space` 里写的 `1274` 只是占位。实际通道数由训练脚本的 `n_fea` 决定，仓库里能看到 **488**（`Resnet34_tf_PPO2_fea488...`）和 **1274**（`n_fea1274/`）两套实验并行存在。

### 算法与超参

主干是 `stable_baselines.PPO2` + 自定义策略网络：

- `sbln_learning/custom_policy/tf/resnet{18,34,50,101}_policy_tf.py` —— TF1 手写 ResNet（`conv2d` 用 `kernel_size=(3,1)`、`data_format="channels_first"`，配合 `C×34` 的牌面张量做横向卷积），实现 `ActorCriticPolicy` 的 `step / proba_step / value`。
- `sbln_learning/custom_policy/keras/` —— Keras 版 ResNet50 策略（早期实验）。
- `sbln_learning/torch_v/` —— PyTorch 复现：`agent/PPO_resnet50.py`、`agent/DQN_resnet50.py`、`policy_model/resnet_50.py`、`replay_buffer.py`、`rl_train.py` / `rl_train_ppo.py`，配置集中在 `rl_config.py`（`state_dim=1330, action_dim=34, actor_lr=1e-5, critic_lr=1e-4, lmbda=0.95, eps=0.2, ppo_epochs=10, gamma=0.99`）。
- `sbln_learning/training_files/train_with_raw/train_ppo2_with_resnet34.py`（以 ResNet34 为例，其余网络同构）：

  | 超参 | 值 | 备注 |
  | --- | --- | --- |
  | `N_STEPS` | 2048 | 每次更新的采样步数 |
  | `NMINIBATCHES` | 2048 // 256 = 8 | |
  | `NOPTEPOCHS` | 8 | |
  | `TOTAL_TIMESTEPS` | 1e8 | |
  | `GAMMA` | 0.925 | 注释说明取 `1 - 1/平均步长` 再微调 |
  | `ENT_COEF` | 0.01 | 新实验降到 0.001 ~ 0.009 |
  | `VF_COEF` | 0.5 | |
  | `INIT_CLIPRANGE` | 0.2 | |
  | `INIT_LR` | 3e-4 | `LinearSchedule` 线性衰减到 `3e-5` |
  | `n_fea` | 488 / 1274 | 两套特征维 |

### 奖励塑形

对手与奖励配置写在每个训练脚本的 `INFO_ON_LOG` 里，可读出这些开关：

- 对手组成：`["zhiyi", "random", "random"]`（1 个专家 + 2 个随机）；
- `midReward`：中间步奖励系数（如 `0.01`）；
- `sqrtReward: True`：对局终局奖励开方；
- `abort_reward: -1`：流局/中止惩罚；
- `illegaA: -3`：非法动作惩罚；
- `isGang_score: False`：杠分不计入奖励。

奖励的计算落在 `mahjong.py::Mcts_game.final_reward()` 与 `Game_RL.mah_step()` 内的 `score_op()` 分支里（番型 → 分数变更）。

### 专家模型接口

`mah_tool/url_recommend.py` 通过 HTTP 调用作者内网的智一服务拿推荐出牌/操作：

```python
outcard_shangrao_url  = "http://172.81.238.92:8082/shangraoMJ/v6/outcard"
operate_shangrao_url  = "http://172.81.238.92:8082/shangraoMJ/v6/operate"
```

`MahEnv.get_zhiyi_recommend_action()` / `get_url_recommand()` 把返回的十进制牌转成 0–33 索引供训练使用。**这两个地址是私有 IP，外部环境不可达**，需自行替换或改用本地专家。

## 技术栈 / 依赖

`requirements.txt` 实际内容（关键项）：

| 依赖 | 版本 |
| --- | --- |
| `tensorflow-gpu` | 1.15.0 |
| `tensorflow-estimator` | 1.15.1 |
| `stable-baselines` | 2.10.0 |
| `gym` | 0.17.3 |
| `Keras` | 2.3.0 |
| `numpy` | 1.19.4 |
| `scipy` | 1.5.4 |
| `pandas` | 1.1.4 |
| `h5py` | 3.1.0 |
| `matplotlib` | 3.3.3 |
| `opencv-python` | 4.4.0.46 |
| `protobuf` | 3.14.0 |

- Python **3.6**（脚本 shebang 指向作者服务器的 `venvs/scmj_ppo2/bin/python3.6`）。
- PyTorch 部分（`sbln_learning/torch_v/`）不在 `requirements.txt` 里，需自行安装 CUDA 版 torch。
- **必须 Linux x86_64**：`mah_tool/so_lib/*.so` 是 `ELF 64-bit LSB shared object, x86-64`（Ubuntu 16.04 编译，带 debug_info）。`mah_tool/接口说明.txt` 明确写：「由于编译 so 文件时采用的是 ubuntu16.04，故只能在 ubuntu 系统中才能运行」「python 版本最好采用 3.6.5」「python 版本不支持 3.7 及以上」。

## 目录结构

```
rlcode/
├── mahEnv.py                 # gym.Env 封装：动作空间 34、Suphx 特征 + mask
├── mah_state.py              # RL_state：单座位视野快照
├── mahjong.py                # 规则/对局核心（Game2 → Mcts_game → Game_RL → MahjongEnv），约 42 KB
├── mah_player.py             # Player_RL：出牌/操作决策、吃碰杠胡执行、模型加载分支
├── load_model.py             # 旧版训练/评测脚本（路径硬编码，见「已知问题」）
├── main.py                   # PyCharm 模板，非入口
├── requirements.txt
├── mah_tool/
│   ├── tool2.py                        # 十进制/十六进制牌值转换、Suphx 特征入口、掩码
│   ├── feature_extract_v10.py          # 特征提取（v10）
│   ├── shangrao_hu.py                  # 上饶麻将胡牌/番型判定
│   ├── url_recommend.py                # 智一专家 HTTP 接口封装
│   ├── 接口说明.txt                     # so 包说明与运行环境约束
│   ├── suphx_extract_features/         # Suphx 特征实现（向听/进张/多派系）
│   └── so_lib/                         # 编译好的 .so + 同名 Python 包装
│       ├── fan_cal.so / lib_MJ.so / shangraoMJ_v5.so / sr_xt_ph.so
│       ├── srmj_v5/                    # 上饶 v5 规则/搜索树（py2.7 + cython 风格源码）
│       └── srmj_v6/                    # v6 规则与推荐
├── interface/interface_v1/             # 2018 年版 TF1 推荐接口（模型文件为空，见「已知问题」）
│   ├── model.py / feature_extract_v8.py / recommand.py
│   └── save/                           # 20180918_*_noking_100.ckpt.*（全部 0 字节）
└── sbln_learning/
    ├── custom_policy/tf/               # ResNet18/34/50/101 的 ActorCriticPolicy
    ├── custom_policy/keras/            # Keras 版 ResNet50 策略
    ├── torch_v/                        # PyTorch PPO/DQN、replay buffer、rl_config
    └── training_files/                 # 训练脚本 + 训练产物（约 703 MB，占仓库绝大部分）
        ├── train_with_raw/             # train_ppo{1,2}_with_resnet{18,34,50,101}.py
        │   ├── tensorboardLog/         # 44 组 n_fea1274 实验 + 4 组 fea488 实验（约 699 MB）
        │   └── train_log/              # 每次启动一个 SAVETIME-* 目录：log.txt + progress.json
        ├── train_from_saveModel/       # 断点续训
        ├── pretrain/                   # 专家示范预训练
        └── progress.json
```

代码规模：80 个 Python 文件，约 25000 行（不含 `.git`）。

## 快速开始

前提：**Linux x86_64 + NVIDIA GPU + CUDA 10.x + Python 3.6**。macOS / Windows 无法加载 `so_lib` 里的 `.so`。

```bash
# 1. 环境（Python 3.6，与打包 so 的版本对齐）
conda create -n scmj python=3.6.5 -y && conda activate scmj
pip install -r requirements.txt        # 含 tensorflow-gpu==1.15.0 / stable-baselines==2.10.0

# 2. 冒烟测试：环境能否正常 reset/step
python mahEnv.py                       # 文件末尾自带 __main__ 测试块

# 3. 训练（必须在该脚本所在目录执行，脚本内有 sys.path.append('../../../')）
cd sbln_learning/training_files/train_with_raw
python train_ppo2_with_resnet34.py     # 结果写入 ./tensorboardLog 与 ./train_log/SAVETIME-*

# 4. 看曲线
tensorboard --logdir sbln_learning/training_files/train_with_raw/tensorboardLog
```

PyTorch 路线的入口是 `sbln_learning/torch_v/rl_train_ppo.py`，配置在 `torch_v/rl_config.py` —— 其中 `env_test.mahjong.player{1,2,3} = Player_RL(..., "zhiyi_last")` 表示用智一当对手。注意智一地址为私有 IP，需替换后才能跑。

## 已知限制 / TODO

**已知问题（均为实测确认）**

1. `main.py` 是 PyCharm 生成的 `print_hi('PyCharm')` 模板，不是程序入口；仓库也没有 README/CLI，入口只能靠训练脚本。
2. `load_model.py` 顶部硬编码 `sys.path.append('/home/tonnn/.nas/.xiu/works/node4-shangrao_mj_rl_v4_suphx/')`，并 `from reward.extract import ...` —— 仓库内**不存在** `reward` 包（`sbln_learning/torch_v/reward/` 才是），导入即失败。
3. `interface/interface_v1/save/` 下 6 个 `.ckpt.*` 文件**全是 0 字节**（`checkpoint` 索引文件还在，指向 `20180918_sys_noking_100.ckpt`），`interface_v1/recommand.py` 因此无法真正恢复模型。
4. `mahEnv.MahEnv.observation_space` 声明的 `(1274, 34, 1)` 与实际特征通道数不一致（训练中用过 488 与 1274），不能依赖该声明做 shape 校验。
5. `mah_tool/so_lib/*.so` 为 Ubuntu 16.04 编译的 x86-64 ELF，且 `接口说明.txt` 声明**不支持 Python 3.7+**，实际可用环境非常窄。
6. 智一专家接口写死内网地址 `172.81.238.92:8082`，`mah_player.py` 里还写死了 `/home/tonnn/xiu/xy/...` 的 `.pth` 模型路径，离开作者机器即不可用。
7. 仓库体积 **769 MB**，其中 `sbln_learning/training_files/train_with_raw/tensorboardLog` 约 **699 MB**、`train_log` 里每一次启动都留一个 `SAVETIME-*` 目录 —— 建议改 Git LFS 或从仓库剥离。
8. 无 `.gitignore`、无实体 LICENSE 文件，`.idea/` 已入库。
9. `interface/interface_v1/recommand.py` 的动作语义注释提到「新疆麻将没有吃牌」，与上饶麻将规则不符，属复制遗留。

**训练记录**

`train_log/SAVETIME-*/log.txt` 与 `progress.json` 保存了每次运行的完整超参与 PPO 统计量（`approxkl`、`clipfrac`、`explained_variance`、`policy_entropy`、`value_loss`、`ep_win_rate`、`ep_len_mean` 等），`training_files/progress.json` 与 `ppo2_our_entCoef0.005_...png` 是对应的汇总曲线。这些都只是训练过程记录，仓库没有给出任何形式的对局强度对比或基线成绩。

**TODO**

- [ ] 把 `so_lib` 换成可移植方案（重新编译到目标 Python/GLIBC，或提供纯 Python 规则实现），去掉「必须 Ubuntu 16.04 + Py3.6」的硬约束。
- [ ] 把智一接口与本地模型路径改为配置项（env / yaml），提供离线随机对手作为默认。
- [ ] 补回或删除 `interface_v1/save/` 的空 ckpt，避免误导。
- [ ] 清理 `training_files` 下 700 MB 训练产物，只保留汇总曲线与最优模型。
- [ ] 为 `torch_v` 部分单独补 `torch` 依赖与运行说明（当前 `requirements.txt` 未覆盖）。
- [ ] 补 `.gitignore`（`__pycache__/`、`*.so`、`train_log/`）、补 LICENSE。
