# DiCo-NLI (SemEval-2027 Task 2)

Code cho system tham gia [SemEval-2027 Task 2: DiCo-NLI](https://inigolopezgazpio.net/SemEval-2027-Task-2-DiCo-NLI/). Kế hoạch nghiên cứu nằm trong [PLAN.md](PLAN.md).

## Cấu trúc

```
configs/           base.yaml + một file YAML cho mỗi thí nghiệm
src/dico/
  env.py           nhận biết local/Kaggle: đường dẫn, bf16/fp16, seed
  data.py          đọc CSV, nhãn, Rev(), ghép cặp đảo (rev_idx)
  decode.py        argmax | joint | joint_structural
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

| Decoder | Ý nghĩa |
|---|---|
| `argmax` | Dự đoán độc lập cho từng instance |
| `joint` | Với mỗi cặp đảo, chọn l tối đa hoá log p(l\|A,B) + log p(Rev(l)\|B,A). Luôn cho SoftCons = 1 |
| `joint_structural` | **Thử nghiệm.** Dùng artifact của dữ liệu: trong train/dev, chỉ các instance `NEGATIVE_OTHER` là không có cặp đảo. Phải hỏi BTC trước khi dùng cho bài nộp |

## Ghi chú về dữ liệu

- Cặp đảo được ghép bằng cách hoán đổi text trong cùng `pair_id`, không dựa vào hậu tố `original`/`flipped`. Ví dụ ở track 4, `en-es__flipped` là cặp đảo của `es-en__original`.
- Đọc CSV với `keep_default_na=False` để các cụm như "NA" hay "None" không bị pandas biến thành giá trị rỗng.
