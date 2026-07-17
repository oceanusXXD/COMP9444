# DeiT-Tiny 实验记录（2026-07-16）
详细记录在 `outputs/`
## 0. 实验命令与可调参数位置

> 规则：学习率、增强和随机种子采用分阶段实验，不做完整的
> `3 seeds × 2 augmentations × 3 learning rates = 18` 组网格搜索。
> 调参阶段只能查看 validation；选择最终配置后才能查看 test。

### 0.1 参数位置

- 主要实验参数：`train_deit.py` 的 `parse_args()`。
  - `--learning-rate`
  - `--weight-decay`
  - `--dropout`
  - `--label-smoothing`
  - `--augmentation`
  - `--seed`
- 训练预算参数：同样位于 `train_deit.py` 的 `parse_args()`。
  - `--epochs`
  - `--warmup-epochs`
  - `--batch-size`
  - `--patience`
  - `--grad-clip`
  - `--workers`
- Mild/Medium 的具体变换强度：`maize_data.py` 的 `build_transform()`。
- 固定标签、切分名和 CSV 约束：`split_utils.py`。这些不是超参数，不应随实验修改。
- 数据审计参数：`audit_dataset.py` 的 `parse_args()`，不影响模型训练。

所有实验都固定以下基础设置，除非记录中明确说明：

```text
model                  = deit_tiny_patch16_224
pretrained checkpoint  = deit_tiny_patch16_224-a1311bcf.pth
epochs                 = 50
warmup epochs          = 5
batch size             = 16
patience               = 8
weight decay           = 1e-2
dropout                = 0.1
label smoothing        = 0.1
optimizer              = AdamW
scheduler              = warm-up + cosine decay
selection metric       = validation Macro-F1
```

### 0.2 阶段一：使用 seed 42 筛选学习率（3e-5 validation macro-f1 0.7175）

先尝试设计文档建议的三个数量级。除学习率外，其他参数完全相同。
#### 1e-5
```powershell
python train_deit.py --seed 42 --augmentation mild --learning-rate 1e-5 --epochs 50 --warmup-epochs 5 --patience 8 --output-dir outputs/lr_1e-5_mild_seed42
```
| 实际完成 epochs | 21 |
| 最佳 epoch | 13 |
| Early Stopping 是否触发 | YES |
| 最佳 epoch 的 train loss | 0.5749 |
| 最佳 epoch 的 train accuracy | 0.8429 |
| 最佳 epoch 的 validation loss | 0.8323 |
| 最佳 epoch 的 validation macro f1 | 0.6474 |
| 最终 epoch 的 train loss | 0.4626 |
| 最终 epoch 的 train accuracy | 0.9214 |
| 最终 epoch 的 validation loss | 0.8940 |
| 最终 epoch 的 validation macro f1 | 0.6222 |

| 指标 | 总体 | N0 | N75 | NFull |
|---|---:|---:|---:|---:|
| Precision | N/A | 0.7536231884057971 | 0.47619047619047616 | 0.7291666666666666 |
| Recall | N/A | 0.8666666666666667 | 0.5 | 0.5833333333333334 |
| F1 | N/A | 0.8062015503875969 | 0.4878048780487805 | 0.6481481481481481 |
| Support | 180 | 60 | 60 | 60 |

Validation 混淆矩阵（行是真实类别，列是预测类别）：
,N0,N75,NFull
N0,0.8666666666666667,0.13333333333333333,0.0
N75,0.2833333333333333,0.5,0.21666666666666667
NFull,0.0,0.4166666666666667,0.5833333333333334

#### 3e-5
```powershell
python train_deit.py --seed 42 --augmentation mild --learning-rate 3e-5 --epochs 50 --warmup-epochs 5 --patience 8 --output-dir outputs/lr_3e-5_mild_seed42
```
| 实际完成 epochs | 29 |
| 最佳 epoch | 21 |
| Early Stopping 是否触发 | YES |
| 最佳 epoch 的 train loss | 0.3517 |
| 最佳 epoch 的 train accuracy | 0.9750 |
| 最佳 epoch 的 validation loss | 0.9353 |
| 最佳 epoch 的 validation macro f1 | 0.7175 |
| 最终 epoch 的 train loss | 0.3031 |
| 最终 epoch 的 train accuracy | 0.9964 |
| 最终 epoch 的 validation loss | 1.0878 |
| 最终 epoch 的 validation macro f1 | 0.6711 |

| 指标 | 总体 | N0 | N75 | NFull |
|---|---:|---:|---:|---:|
| Precision | N/A | 0.782608695652174 | 0.6 | 0.7678571428571429 |
| Recall | N/A | 0.9 | 0.55 | 0.7166666666666667 |
| F1 | N/A | 0.8372093023255814 | 0.5739130434782609 | 0.7413793103448276 |
| Support | 180 | 60 | 60 | 60 |

Validation 混淆矩阵（行是真实类别，列是预测类别）：
,N0,N75,NFull
N0,0.9,0.1,0.0
N75,0.23333333333333334,0.55,0.21666666666666667
NFull,0.016666666666666666,0.26666666666666666,0.7166666666666667

#### 1e-4
```powershell
python train_deit.py --seed 42 --augmentation mild --learning-rate 1e-4 --epochs 50 --warmup-epochs 5 --patience 8 --output-dir outputs/lr_1e-4_mild_seed42
```
| 实际完成 epochs | 20 |
| 最佳 epoch | 12 |
| Early Stopping 是否触发 | YES |
| 最佳 epoch 的 train loss | 0.4601 |
| 最佳 epoch 的 train accuracy | 0.9000 |
| 最佳 epoch 的 validation loss | 0.9258 |
| 最佳 epoch 的 validation macro f1 | 0.6677 |
| 最终 epoch 的 train loss | 0.3358 |
| 最终 epoch 的 train accuracy | 0.9762 |
| 最终 epoch 的 validation loss | 1.1050 |
| 最终 epoch 的 validation macro f1 | 0.6009 |

| 指标 | 总体 | N0 | N75 | NFull |
|---|---:|---:|---:|---:|
| Precision | N/A | 0.7884615384615384 | 0.5066666666666667 | 0.7547169811320755 |
| Recall | N/A | 0.6833333333333333 | 0.6333333333333333 | 0.6666666666666666 |
| F1 | N/A | 0.7321428571428571 | 0.562962962962963 | 0.7079646017699115 |
| Support | 180 | 60 | 60 | 60 |

Validation 混淆矩阵（行是真实类别，列是预测类别）：
,N0,N75,NFull
N0,0.6833333333333333,0.3,0.016666666666666666
N75,0.16666666666666666,0.6333333333333333,0.2
NFull,0.016666666666666666,0.31666666666666665,0.6666666666666666

#### 2e-5
```powershell
python train_deit.py --seed 42 --augmentation mild --learning-rate 2e-5 --epochs 50 --warmup-epochs 5 --patience 8 --output-dir outputs/lr_2e-5_mild_seed42
```
| 实际完成 epochs | 40 |
| 最佳 epoch | 32 |
| Early Stopping 是否触发 | YES |
| 最佳 epoch 的 train loss | 0.3016 |
| 最佳 epoch 的 train accuracy | 0.9952 |
| 最佳 epoch 的 validation loss | 1.0198 |
| 最佳 epoch 的 validation macro f1 | 0.7054 |
| 最终 epoch 的 train loss | 0.2992 |
| 最终 epoch 的 train accuracy | 0.9976 |
| 最终 epoch 的 validation loss | 1.0432 |
| 最终 epoch 的 validation macro f1 | 0.6914 |

| 指标 | 总体 | N0 | N75 | NFull |
|---|---:|---:|---:|---:|
| Precision | N/A | 0.847457627118644 | 0.559322033898305 | 0.7096774193548387 |
| Recall | N/A | 0.8333333333333334 | 0.55 | 0.7333333333333333 |
| F1 | N/A | 0.8403361344537815 | 0.5546218487394958 | 0.7213114754098361 |
| Support | 180 | 60 | 60 | 60 |

Validation 混淆矩阵（行是真实类别，列是预测类别）：
,N0,N75,NFull
N0,0.8333333333333334,0.16666666666666666,0.0
N75,0.15,0.55,0.3
NFull,0.0,0.26666666666666666,0.7333333333333333

#### 6e-5
```powershell
python train_deit.py --seed 42 --augmentation mild --learning-rate 6e-5 --epochs 50 --warmup-epochs 5 --patience 8 --output-dir outputs/lr_6e-5_mild_seed42
```
| 实际完成 epochs | 24 |
| 最佳 epoch | 16 |
| Early Stopping 是否触发 | YES |
| 最佳 epoch 的 train loss | 0.3838 |
| 最佳 epoch 的 train accuracy | 0.9536 |
| 最佳 epoch 的 validation loss | 0.8402 |
| 最佳 epoch 的 validation macro f1 | 0.6956 |
| 最终 epoch 的 train loss | 0.3253 |
| 最终 epoch 的 train accuracy | 0.9833 |
| 最终 epoch 的 validation loss | 1.0783 |
| 最终 epoch 的 validation macro f1 | 0.6650 |

| 指标 | 总体 | N0 | N75 | NFull |
|---|---:|---:|---:|---:|
| Precision | N/A | 0.782608695652174 | 0.5689655172413793 | 0.7358490566037735 |
| Recall | N/A | 0.9 | 0.55 | 0.65 |
| F1 | N/A | 0.8372093023255814 | 0.559322033898305 | 0.6902654867256637 |
| Support | 180 | 60 | 60 | 60 |

Validation 混淆矩阵（行是真实类别，列是预测类别）：
,N0,N75,NFull
N0,0.9,0.08333333333333333,0.016666666666666666
N75,0.23333333333333334,0.55,0.21666666666666667
NFull,0.016666666666666666,0.3333333333333333,0.65

选择规则：validation Macro-F1 最高者优先；如果非常接近，依次比较 N75 F1、
validation loss、曲线稳定性和更早达到最佳 epoch 的配置。

### 0.3 阶段二：比较 Mild 与 Medium augmentation（决定使用mild）

Mild 结果直接复用阶段一最佳学习率对应的实验，只额外运行一次 Medium。

```powershell
python train_deit.py --seed 42 --augmentation medium --learning-rate 3e-5 --epochs 50 --warmup-epochs 5 --patience 8 --output-dir "outputs/aug_medium_lr_3e-5_seed42"
```
| 实际完成 epochs | 50 |
| 最佳 epoch | 42 |
| Early Stopping 是否触发 | YES |
| 最佳 epoch 的 train loss | 0.3069 |
| 最佳 epoch 的 train accuracy | 0.9917 |
| 最佳 epoch 的 validation loss | 0.9651 |
| 最佳 epoch 的 validation macro f1 | 0.7240 |
| 最终 epoch 的 train loss | 0.3071 |
| 最终 epoch 的 train accuracy | 0.9929 |
| 最终 epoch 的 validation loss | 1.0193 |
| 最终 epoch 的 validation macro f1 | 0.7012 |

| 指标 | 总体 | N0 | N75 | NFull |
|---|---:|---:|---:|---:|
| Precision | N/A | 0.8181818181818182 | 0.6 | 0.7571428571428571 |
| Recall | N/A | 0.75 | 0.55 | 0.8833333333333333 |
| F1 | N/A | 0.782608695652174 | 0.5739130434782609 | 0.8153846153846154 |
| Support | 180 | 60 | 60 | 60 |

Validation 混淆矩阵（行是真实类别，列是预测类别）：
,N0,N75,NFull
N0,0.75,0.25,0.0
N75,0.16666666666666666,0.55,0.2833333333333333
NFull,0.0,0.11666666666666667,0.8833333333333333

选择规则仍以 validation Macro-F1 为主
对比两个类型的混淆矩阵，发现 medium 只是多判断正确了一幅图，考虑到 mild 更简单且收敛速度更快，因此依旧考虑 mild

### 0.4 阶段三：固定配置并补齐三个随机种子

seed 42 的结果可以直接用对应实验，补跑 `22` 和 `62`。
#### seed 22
```powershell
python train_deit.py --seed 22 --augmentation mild --learning-rate 3e-5 --epochs 50 --warmup-epochs 5 --patience 8 --output-dir "outputs/final_mild_lr_3e-5_seed22"
```
| 实际完成 epochs | 24 |
| 最佳 epoch | 16 |
| Early Stopping 是否触发 | YES |
| 最佳 epoch 的 train loss | 0.4296 |
| 最佳 epoch 的 train accuracy | 0.9381 |
| 最佳 epoch 的 validation loss | 0.8967 |
| 最佳 epoch 的 validation macro f1 | 0.7201 |
| 最终 epoch 的 train loss | 0.3222 |
| 最终 epoch 的 train accuracy | 0.9845 |
| 最终 epoch 的 validation loss | 1.0294 |
| 最终 epoch 的 validation macro f1 | 0.6686 |

| 指标 | 总体 | N0 | N75 | NFull |
|---|---:|---:|---:|---:|
| Precision | N/A | 0.8 | 0.603448275862069 | 0.7543859649122807 |
| Recall | N/A | 0.8666666666666667 | 0.5833333333333334 | 0.7166666666666667 |
| F1 | N/A | 0.832 | 0.5932203389830508 | 0.7350427350427351 |
| Support | 180 | 60 | 60 | 60 |

Validation 混淆矩阵（行是真实类别，列是预测类别）：
,N0,N75,NFull
N0,0.8666666666666667,0.13333333333333333,0.0
N75,0.18333333333333332,0.5833333333333334,0.23333333333333334
NFull,0.03333333333333333,0.25,0.7166666666666667

#### seed 62
```powershell
python train_deit.py --seed 62 --augmentation mild --learning-rate 3e-5 --epochs 50 --warmup-epochs 5 --patience 8 --output-dir "outputs/final_mild_lr_3e-5_seed62"
```
| 实际完成 epochs | 25 |
| 最佳 epoch | 17 |
| Early Stopping 是否触发 | YES |
| 最佳 epoch 的 train loss | 0.3991 |
| 最佳 epoch 的 train accuracy | 0.9464 |
| 最佳 epoch 的 validation loss | 0.9504 |
| 最佳 epoch 的 validation macro f1 | 0.6826 |
| 最终 epoch 的 train loss | 0.3221 |
| 最终 epoch 的 train accuracy | 0.9833 |
| 最终 epoch 的 validation loss | 1.2994 |
| 最终 epoch 的 validation macro f1 | 0.5905 |

| 指标 | 总体 | N0 | N75 | NFull |
|---|---:|---:|---:|---:|
| Precision | N/A | 0.746031746031746 | 0.5423728813559322 | 0.7586206896551724 |
| Recall | N/A | 0.7833333333333333 | 0.5333333333333333 | 0.7333333333333333 |
| F1 | N/A | 0.7642276422764228 | 0.5378151260504201 | 0.7457627118644068 |
| Support | 180 | 60 | 60 | 60 |

Validation 混淆矩阵（行是真实类别，列是预测类别）：
,N0,N75,NFull
N0,0.7833333333333333,0.21666666666666667,0.0
N75,0.23333333333333334,0.5333333333333333,0.23333333333333334
NFull,0.03333333333333333,0.23333333333333334,0.7333333333333333

### 0.5 阶段四：配置锁定后评估 Test

对三个种子的最佳 checkpoint 分别执行一次。`$SEED42_DIR` 必须指向阶段一或
阶段二中被选为最终配置的 seed 42 输出目录。
#### seed 42 test
```powershell
python train_deit.py --evaluate-only --checkpoint "outputs/lr_3e-5_mild_seed42/best_checkpoint.pt" --eval-split test --output-dir "outputs/lr_3e-5_mild_seed42"
```
label   N0  N75  NFull
split
test    60   60     60
train  280  280    280
val     60   60     60
test: accuracy=0.7444, balanced_accuracy=0.7444, macro_f1=0.7430, level_mae=0.2556

#### seed 22 test
```powershell
python train_deit.py --evaluate-only --checkpoint "outputs/final_mild_lr_3e-5_seed22/best_checkpoint.pt" --eval-split test --output-dir "outputs/final_mild_lr_3e-5_seed22"
```
label   N0  N75  NFull
split
test    60   60     60
train  280  280    280
val     60   60     60
test: accuracy=0.7056, balanced_accuracy=0.7056, macro_f1=0.7032, level_mae=0.2944

#### seed 62 test
```powershell
python train_deit.py --evaluate-only --checkpoint "outputs/final_mild_lr_3e-5_seed62/best_checkpoint.pt" --eval-split test --output-dir "outputs/final_mild_lr_3e-5_seed62"
```
label   N0  N75  NFull
split
test    60   60     60
train  280  280    280
val     60   60     60
test: accuracy=0.7389, balanced_accuracy=0.7389, macro_f1=0.7383, level_mae=0.2667

## 1. 结果汇总
最终配置固定为 DeiT-Tiny、Mild augmentation、学习率 `3e-5`、weight decay
`1e-2`、dropout `0.1` 和 label smoothing `0.1`。三个随机种子的 Test 结果如下：

| Seed | Accuracy | Macro-F1 | N0 F1 | N75 F1 | NFull F1 | Level MAE |
|---:|---:|---:|---:|---:|---:|---:|
| 22 | 0.7056 | 0.7032 | 0.8293 | 0.5470 | 0.7333 | 0.2944 |
| 42 | 0.7444 | 0.7430 | 0.8689 | 0.6102 | 0.7500 | 0.2556 |
| 62 | 0.7389 | 0.7383 | 0.8780 | 0.6290 | 0.7080 | 0.2667 |
| Mean ± Std | 0.7296 ± 0.0210 | 0.7282 ± 0.0218 | 0.8587 ± 0.0259 | 0.5954 ± 0.0430 | 0.7304 ± 0.0212 | 0.2722 ± 0.0200 |

整体 `N0` 的识别效果最好，`N75` 的平均 F1 最低且标准差最大，说明中间等级仍是主要困难。训练后期普遍出现 train loss 继续下降而 validation loss 不服从 train 曲线反而上扬的现象，按 validation Macro-F1 保存最佳 checkpoint，并使用 Early Stopping 是合理的。综合性能、收敛速度和实验复杂度，最终保留 `3e-5 + Mild augmentation` 作为 DeiT-Tiny baseline 配置。