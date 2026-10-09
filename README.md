# RGBtoSPAD

用一张 RGB 照片 + 少量标定点，估算叶片叶绿素 **SPAD** 值的桌面工具。

![软件界面](软件界面图.png)

## 功能

- 拖框选取标定区域，输入实测 SPAD 值建立回归模型
- 双模型对比：多元线性回归 / Elastic Net，实时给出方程、R²、MAE
- 区域选择、网格划分（1–50 × 1–50）、逐格 SPAD 预测
- 生成网格预览图与 SPAD 热力图，结果导出 Excel + PNG

## 环境

Python 3.9+（依赖 Tkinter，随官方安装包自带；Linux 需 `sudo apt install python3-tk`）

```bash
pip install -r requirements.txt
```

## 使用

```bash
python RGBtoSPAD_v2.py
```

1. **加载图片**：选图 →【加载】，叶片摆不正可用旋转滑杆调整。
2. **① 选择区域**：拖框选出只包含叶片的矩形区域。
3. **② 校准模式**：在颜色均匀处拖框取样，填入该点的实测 SPAD 值。至少 3 个点，
   建议 5–8 个且 SPAD 值跨度尽量大，完成后点【完成校准并拟合】。
4. **③ 划分网格 → ④ 预测 SPAD**：输入行列数后预测，用【网格预览】/【热力图】切换查看。
5. **⑤ 保存结果**：导出 Excel 与两张图片。

热力图配色：**红 = 低 SPAD，深绿 = 高 SPAD**。颜色按当前叶片 Min–Max 归一化，
**跨图片不可比较，请比数值**。

## 拍摄要求

标定照片与待测照片必须在**同一光源、同一白平衡、同曝光**下拍摄（建议手动锁定白平衡、关闭 HDR）。
"颜色 ↔ SPAD"的映射只在相同成像条件下成立，这是精度的决定性因素。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `xxx.xlsx` | SPAD预测结果 / 模型对比 / Elastic Net参数 / 过滤设置 |
| `xxx_网格预览.png` | 带编号的网格划分图 |
| `xxx_热力图.png` | SPAD 分布热力图 |

## 已知问题

**绿色背景过滤对非绿色像素失效。** `calculate_rgb_with_filter()` 用 uint8 数组相减判断绿色度，
负值会按 256 回绕（如 `120 − 140 = 236`），导致 `G < max(R, B)` 的土壤、灰背景像素被误判为叶肉保留，
被排除的反而是浅色黄化叶片。修复方式是比较前提升为有符号整型：

```python
roi_reshaped = roi.reshape(-1, 3).astype(np.int16)
green_mask = (roi_reshaped[:, 1] - np.maximum(roi_reshaped[:, 0], roi_reshaped[:, 2])) > green_threshold
```

不勾选该选项时不受影响。

另外，程序给出的 R² 是**训练集**拟合优度，不代表泛化精度。建议预留 1–2 个已知 SPAD 的点不参与拟合，
用它们验证后再用于正式数据。

## 许可

[MIT](LICENSE)
