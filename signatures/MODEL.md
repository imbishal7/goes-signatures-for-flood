# June same-day ConvLSTM — model overview

**Task.** From one **CST day**'s GOES imagery, predict that day's NWS flood-warning grid.
Defined in [`notebooks/convlstm_june_train.py`](notebooks/convlstm_june_train.py); orchestrated by
[`notebooks/train_convlstm_june.ipynb`](notebooks/train_convlstm_june.ipynb). `B` = batch per GPU.

Grid: 50 km CONUS-land, **R×C = 59×95** (5,605 cells, **3,360 over land** — loss/metrics on land only).

## Inputs

| input | shape | notes |
|---|---|---|
| GOES cube `x` | `(B, 24, 16, 1500, 2500)` f16 | 24 hourly CST frames × 16 ABI bands, full-res ABI grid |
| CST-hour `t` | `(B, 24)` | local hour (0–23) of each frame |
| **GLM map** `glm` *(optional)* | `(B, 3, 59, 95)` | per-cell daily log[count, energy, area] on the grid |

On the GPU the hour is broadcast to a channel and concatenated → model input **`(B, 24, 17, 1500, 2500)`** (`N_CH = 17`).

## Pipeline (middle stages, with shapes)

```
x (B, 24, 17, 1500, 2500)
   │  Encoder  (per-frame CNN, shared weights, gradient-checkpointed) — applied to each of the 24 frames
   │    conv_block 17→32 → MaxPool/2 → conv_block 32→64 → conv_block 64→64
   ▼
per-frame features  (B, 24, 64, 750, 1250)          # spatial /2  (POOL_STRIDE=2)
   │  ConvLSTM  (one ConvLSTMCell walked over the 24 frames, hidden=64, checkpointed per step)
   ▼
final hidden h  (B, 64, 750, 1250)
   │  CellPool  (scatter-mean encoder pixels → 50 km cells via fixed pixel→cell index)
   ▼
cell features  (B, 64, 59, 95)
   │                                  ┌─ images-only:  (B, 64, 59, 95)
   │  [+ GLM concat if enabled] ──────┤
   │                                  └─ GOES+GLM:     (B, 64+3=67, 59, 95)
   ▼
head  conv_block(C→64) → Conv2d(64→1)
   ▼
logits  (B, 59, 95)   →  sigmoid  →  P(flood) per cell
```

## Output

| output | shape | notes |
|---|---|---|
| per-cell logits | `(B, 59, 95)` | `sigmoid` → P(flood); scored on the 3,360 land cells |
| target `y` | `(B, 59, 95)` 0/1 | warned cells that CST day |

**Loss:** recall-favouring soft **Tversky** (α=0.3, β=0.7) over the land cells.

## With vs. without GLM — the only difference

| | images-only (default) | GOES + GLM (`--glm`) |
|---|---|---|
| extra input | — | GLM `(B, 3, 59, 95)` per-cell map |
| fusion point | — | concat to pooled features at the **cell grid**, before the head |
| head input channels | 64 | 67 |
| learnable params | **513,441** | **515,169** (+1,728) |

Everything upstream of `CellPool` (the GOES Encoder→ConvLSTM path) is **identical** in both modes;
GLM only adds 3 channels at the cell grid just before the head. There is **no climatology or
location prior** — the per-cell prediction comes purely from the imagery (and GLM, if enabled).

## On-disk cache (per CST day, `signatures/cache/convlstm_june/`)

| file | shape / dtype |
|---|---|
| `{day}_x.npy` | `(24, 16, 1500, 2500)` f16  (~2.9 GB) |
| `{day}_t.npy` | `(24,)` f32 — CST hours |
| `{day}_y.npy` | `(59, 95)` uint8 — warning grid |
| `{day}_glm.npy` *(if `--glm`)* | `(3, 59, 95)` f32 — log count/energy/area |
