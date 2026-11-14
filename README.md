# DiCo-NLI (SemEval-2027 Task 2)

Code cho system tham gia [SemEval-2027 Task 2: DiCo-NLI](https://inigolopezgazpio.net/SemEval-2027-Task-2-DiCo-NLI/). Kế hoạch nghiên cứu nằm trong [PLAN.md](PLAN.md).

## Cấu trúc

```
configs/           base.yaml + một file YAML cho mỗi thí nghiệm
src/dico/
  env.py           nhận biết local/Kaggle: đường dẫn, bf16/fp16, seed
  data.py          đọc CSV, nhãn, Rev(), ghép cặp đảo (rev_idx)
  decode.py        argmax | self_swap (an toàn) | joint | joint_structural | graph_*
  evaluate.py      gọi scorer chính thức của BTC
  baselines.py     majority, rule, surface_lr (không cần GPU)
  train.py         fine-tune cross-encoder (M0) và chấm điểm trên dev
  predict.py       dự đoán file bất kỳ (test) bằng model đã lưu, hỗ trợ ensemble
scripts/           setup_data.sh, run_seeds.sh, collect_results.py
kaggle/runner.ipynb   notebook mỏng để chạy trên Kaggle
external/          (gitignored) repo của BTC: data và scorer
runs/              (gitignored) output của từng run
results/           summary.csv: bảng kết quả, có commit
```

## Cài đặt local (Windows, RTX 5060)

```bash
py -3.12 -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -e .
bash scripts/setup_data.sh
```

GPU dòng 50xx (Blackwell) cần PyTorch bản cu128 trở lên. Trên Linux/WSL, thay `.venv/Scripts/python` bằng `.venv/bin/python`.

## Chạy

```bash
# Baseline không dùng GPU (vài giây)
python -m dico.baselines --tracks 1 2 3 4

# M0, một seed
python -m dico.train --config configs/m0_deberta_base_en.yaml --seed 1

# Ghi đè config ngay trên dòng lệnh
python -m dico.train --config configs/m0_deberta_base_en.yaml --seed 1 --set epochs=3 lr=3e-5

# 3 seed, sau đó gom kết quả vào results/summary.csv
PYTHON=.venv/Scripts/python bash scripts/run_seeds.sh configs/m0_deberta_base_en.yaml
```

Mỗi run ghi vào `runs/<name>/seed<k>/` các file sau:
- `config.yaml`
- `dev_track*_probs.npy`
- file dự đoán của từng decoder
- `scores_*/scores.json` (từ scorer chính thức)
- `results.json`

## Kaggle

1. Push repo này lên GitHub.
2. Tạo notebook từ [kaggle/runner.ipynb](kaggle/runner.ipynb), sửa `REPO_URL`, bật GPU và Internet.
3. Chạy hết các cell, tải `runs.zip` về, giải nén vào `runs/` ở local, rồi chạy `python scripts/collect_results.py`.

Đường dẫn và precision được `env.py` tự xử lý: chọn fp16 trên T4, bf16 trên RTX 5060.

## Decoder

Bài nộp chỉ dùng decoder **an toàn**: output cho một instance chỉ phụ thuộc vào text của chính nó ([PLAN.md](PLAN.md) mục 2c). `predict.py` mặc định là `self_swap` và từ chối decoder không an toàn nếu không có `--allow_unsafe`.

| Decoder | An toàn | Ý nghĩa |
|---|---|---|
| `argmax` | Có | Dự đoán độc lập cho từng instance |
| `self_swap` | Có | **Mặc định cho test.** Tự chạy model trên (B, A), chọn l tối đa hoá log p(l\|A,B) + log p(Rev(l)\|B,A) + `neg_bias`·[l = NEG]. Nhất quán khi đảo chiều theo thiết kế. Tune `neg_bias` trên train-OOF bằng `scripts/tune_neg_bias.py` |
| `joint` | Không | Giống `self_swap` nhưng tra twin trong file. Chỉ để so sánh |
| `joint_structural` | Không | Dùng artifact: trong train/dev, chỉ `NEGATIVE_OTHER` là không có twin. Chỉ để phân tích |
| `graph_soft` | Không | Hard reversal + transitivity trên tam giác quan sát, phạt mỗi vi phạm bằng `graph_lambda` |
| `graph_hard` | Không | Hard reversal + hard transitivity trên tam giác quan sát |

Lưu ý: scorer coi NEG dự đoán trên một cặp đảo là không nhất quán, nên `self_swap`/`joint` vẫn có SoftCons < 1 nếu model đoán NEG cho cặp có twin.

Graph dùng định danh node chính xác `(language, phrase)` và không tự sinh latent edge.
Chạy `python scripts/audit_graph.py` trước khi diễn giải kết quả; nếu
`observed_paths = 0`, hai graph decoder rút gọn chính xác về `joint`.

Tune temperature và soft penalty để dùng cho test sau này bằng:

```bash
python scripts/graph_ablation.py --run runs/<name>/seed1 --track 1
```

Script đánh dấu kết quả là dev-tuned; không coi các điểm đó là held-out estimate.

## Ghi chú về dữ liệu

- Cặp đảo được ghép bằng cách hoán đổi text trong cùng `pair_id`, không dựa vào hậu tố `original`/`flipped`. Ví dụ ở track 4, `en-es__flipped` là cặp đảo của `es-en__original`.
- Đọc CSV với `keep_default_na=False` để các cụm như "NA" hay "None" không bị pandas biến thành giá trị rỗng.
