# SemEval-2027 Task 2 (DiCo-NLI): Kế hoạch

Nguồn: [trang task](https://inigolopezgazpio.net/SemEval-2027-Task-2-DiCo-NLI/) · [GitHub](https://github.com/ilopezgazpio/SemEval-2027-Task-2-DiCo-NLI) · [CodaBench](https://www.codabench.org/competitions/18038/)

## 1. Bài toán

- **Input:** cặp cụm từ có thứ tự (A, B), lấy từ PhrasIS (caption ảnh và tiêu đề báo).
- **Output:** 1 trong 4 nhãn:

| Nhãn | Ý nghĩa | Khi đảo thành (B, A) |
|---|---|---|
| `EQUIVALENCE` | A ⇔ B | `EQUIVALENCE` |
| `FORWARD_ENTAILMENT` | A ⇒ B | `BACKWARD_ENTAILMENT` |
| `BACKWARD_ENTAILMENT` | B ⇒ A | `FORWARD_ENTAILMENT` |
| `NEGATIVE_OTHER` | các quan hệ còn lại | `NEGATIVE_OTHER` |

- **Track:** EN, ES, EU (mỗi track 3,042 cặp train và 660 dev); Mixed (18,252 train, 3,960 dev; 18,252 = 3,042 × 6 tổ hợp ngôn ngữ).
- **Phân bố nhãn (train EN):** BWD 881, EQ 802, FWD 881, NEG 478.
- **File nộp:** `instance_id,label`.

## 2. Metrics

| Metric | Đo cái gì | Cần gold? |
|---|---|---|
| Weighted F1 | Độ đúng trên cả 4 nhãn | Có |
| SoftCons | f(A,B) = Rev(f(B,A)): có tự mâu thuẫn không | Không |
| HardCons | Đúng **cả hai chiều** của một cặp | Có |

Mốc pilot (EN, DeBERTa-v3-base): F1 0.75, SoftCons 0.84, HardCons 0.79.

**Nhận xét then chốt:** SoftCons có thể đạt 1.0 chỉ bằng cách thiết kế model (xem M1). Các đội làm tốt sẽ đều đạt mức này, nên SoftCons không tạo khác biệt giữa các đội. Khác biệt nằm ở F1 và HardCons. Khi đã nhất quán theo thiết kế, HardCons gần như bằng độ chính xác tính theo cặp.

## 2b. Phát hiện từ dữ liệu và baseline (9/11/2026)

**Về cấu trúc dữ liệu:**
- Train và dev **đã có sẵn cả hai chiều** (`original`/`flipped`), nên reversal augmentation không thêm được gì.
- Cặp đảo ghép được bằng cách hoán đổi text trong cùng `pair_id`. Kết quả khớp 100% với `reverse_pair_id` của BTC trên mọi track.
- **Artifact:** trong train/dev, mọi instance `NEGATIVE_OTHER` (và chỉ chúng) **không có cặp đảo**. Nếu test giữ cấu trúc này thì nhận diện được NEG một cách hoàn hảo mà không cần model. Cần hỏi hoặc báo BTC; đây cũng là kết quả cho N1.
- **Scorer:** NEG dự đoán trên một cặp đảo luôn bị tính là không nhất quán (SoftCons).

**Baseline trên dev** (`python -m dico.baselines`). Weighted F1 / SoftCons / HardCons:

| Hệ thống | T1 (EN) | T2 (ES) | T3 (EU) | T4 (Mixed) |
|---|---|---|---|---|
| Majority | 0.13 / 0 / 0 | 0.13 / 0 / 0 | 0.13 / 0 / 0 | 0.13 / 0 / 0 |
| Rule (token subset) | 0.39 / 0.33 / 0.29 | 0.40 / 0.38 / 0.32 | 0.32 / 0.26 / 0.21 | 0.08 / 0.03 / 0.02 |
| Surface LR (12 feature độ dài/overlap), joint | 0.55 / 0.98 / 0.65 | 0.53 / 0.98 / 0.64 | 0.51 / 1.00 / 0.66 | 0.42 / 1.00 / 0.58 |
| Surface LR + structural NEG | **0.71** / 1.00 / 0.66 | 0.70 / 1.00 / 0.64 | 0.71 / 1.00 / 0.66 | 0.62 / 1.00 / 0.58 |
| *Pilot DeBERTa-v3-base (LREC)* | *0.75 / 0.84 / 0.79* | | | |
| M0 DeBERTa-v3-base, argmax (3 seed) | 0.824±.011 / 0.886 / 0.836 | | | |
| M0 + joint | 0.819±.016 / 0.921 / 0.852 | | | |
| M0 + joint structural | 0.919±.016 / 1.00 / 0.903 | | | |

M0 của mình vượt pilot (F1 0.82 so với 0.75), train mất chưa đến 1 phút trên RTX 5060. Joint decoding đổi khoảng 0.005 F1 lấy +0.016 HardCons.

**Ý nghĩa cho N1:** một logistic regression chỉ dùng độ dài và overlap, cộng với artifact NEG, đạt F1 0.71 so với 0.75 của DeBERTa trong pilot. HardCons còn cách khoảng 0.13, và đó là phần thật sự cần đến ngữ nghĩa.

## 2c. Giả định luật chơi: chọn cách an toàn nhất (14/11/2026, không hỏi BTC)

Căn cứ README của BTC: EN/ES/EU là "independent monolingual tracks" (dòng 150); external resource được phép nếu ghi rõ trong paper (dòng 327); không dùng nhãn test (dòng 333).

**Quy tắc cho mọi bài nộp test:** dự đoán cho một instance chỉ được phụ thuộc vào **chính text của instance đó** (cộng với model và resource bên ngoài đã train sẵn).

| Kỹ thuật | Trạng thái | Lý do / thay thế |
|---|---|---|
| `joint_structural` (NEG khi không có twin) | **Cấm** | Khai thác cách BTC dựng file. Chỉ báo cáo như phân tích artifact (N1) |
| `joint` / `graph_*` tra twin trong file test | **Thay bằng self-swap** | Tự chạy model trên (B, A) rồi decode argmax log p(l\|A,B) + log p(Rev(l)\|B,A). Kết quả giống `joint` trên dev nhưng không cần biết file có twin hay không |
| Pool theo `pair_id` xuyên track / xuyên instance (`pool_views`, `learned_pool`, `evaluate_oof_pool`, `dynamic_gate`) | **Cấm ở test** | Dùng instance khác và parse `instance_id`. Thay bằng view tự sinh: dịch máy (A, B) sang ngôn ngữ khác rồi pool trên chính instance đó |
| Train trên train/dev của mọi track, loss nhất quán theo `pair_id` trên train | Cho phép | Dữ liệu public có nhãn; ghi rõ trong paper |
| Model pre-train NLI (MNLI/XNLI), MT (NLLB), LLM | Cho phép | Ghi rõ trong metadata và paper |
| Chọn hyper-parameter / ensemble trên dev | Tránh | Tune trên train-OOF group theo `pair_id`; dev chỉ dùng làm held-out |

Hệ quả: mọi con số "structural" và "pooled" ở trên **không còn là số của hệ thống nộp**. Mốc an toàn hiện tại là các dòng `joint` (≈ self-swap).

**Luật go/no-go cho N2 (view dịch máy), chốt ngày 14/11 trước khi xem kết quả** (`scripts/mt_views_eval.py`, track EU là track chính):

| Kết quả `pool:orig+mt` − `orig` | Quyết định |
|---|---|
| Δ HardCons ≥ +1 điểm, khoảng tin cậy 95% (paired bootstrap theo `pair_id`) không chứa 0, và F1 giảm không quá 1 điểm | Go |
| Δ > 0 nhưng khoảng tin cậy chứa 0 | Train thêm 2 seed có lưu model; go nếu cả 3 seed đều dương |
| Δ ≤ 0 | No-go cho phương pháp; giữ làm phân tích "dịch máy làm mất quan hệ có hướng" |

Báo cáo kèm cận trên `pool:orig+gold` (chỉ để phân tích) và tỉ lệ thu hồi (mt − orig) / (gold − orig).

**Kết quả lần 1 (NLLB-200-distilled-600M, 1 seed, track EU):** Δ HardCons = 0.0 [−3.8, +3.7], Δ F1 = −0.6. **No-go.** Cận trên gold: +7.6 HardCons [+4.2, +11.1], +5.5 F1. Tỉ lệ thu hồi 0%.
**Lần 2 là lần cuối** (chốt trước khi chạy): NLLB-200-distilled-1.3B, cùng luật, cùng model NLI, cùng bias. Kết quả thế nào cũng không thử thêm model dịch nào nữa.
**Kết quả lần 2 (NLLB-1.3B, EU):** pool:orig+mt Δ HardCons −0.7 [−4.5, +2.9], Δ F1 −3.0; orig+mt+mix tệ hơn rõ (Δ HardCons [−7.8, −0.4]). **No-go, đóng hướng dịch máy.** Cận trên gold vẫn +7.6 HardCons: thông tin nằm trong bản dịch chuẩn, MT không tái tạo được.

**Phương pháp 1: consistency xuyên ngôn ngữ khi train** (`configs/xling_js_all.yaml`, `configs/xling_en_all.yaml`; chốt ngày 14/11 trước khi chạy). Dùng bản dịch chuẩn có sẵn trong train, không cần dịch máy; lúc test vẫn chỉ đọc một instance.
- So với `m0_mdeberta_xnli_pair_all` (cùng model, cùng `pair_consistency_weight`, cùng bias −10, `self_swap`), mỗi bên 3 seed. Trọng số `xling_consistency_weight = 0.5` chốt trước, không tune trên dev.
- Thành công nếu: HardCons trung bình 3 seed trên EU tăng ≥ +1 điểm, F1 không giảm quá 1 điểm, và tỉ lệ bất đồng giữa EN/ES/EU trên dev giảm so với mức 25% hiện tại. Báo cáo cả 4 track cùng paired bootstrap trên ensemble 3 seed.
- Nếu `js` và `en` cùng đạt, chọn bản có trung bình 3 metric trên train-OOF cao hơn, không chọn theo dev.
- **Kết quả bản thử nhanh** (mono, 1 seed): không có tín hiệu; bất đồng xuyên ngôn ngữ còn tăng (22.6% → 24–26%). Giả thuyết: trên train, các ngôn ngữ đã chung nhãn nên loss nhất quán gần như bằng 0. Tạm dừng.

**Ý 1: đầu entailment hai chiều** (`method: bientail`, `configs/bientail_all.yaml`; chốt ngày 14/11 trước khi chạy).
- Probe zero-shot (không train): EN F1 0.63 với luật bảng 4 ô, 0.66 với một LR nhỏ trên 6 log-prob.
- So với đầu 4 nhãn ở cùng dữ liệu và lịch train: `bientail_mono` vs `quick_base`, `bientail_all` vs `m0_mdeberta_xnli_pair_all`; 3 seed mỗi bên; so ở bias 0 và ở bias chọn trên train-OOF (argmax trung bình 3 metric).
- Thành công nếu HardCons trung bình 3 seed tăng ≥ +1 điểm ở ít nhất 2/3 track mono, F1 không giảm quá 1 điểm ở track nào.
- Kèm control `full_1ep` (4 track, 1 epoch, cùng số bước với bản mono) để tách hiệu ứng "bỏ Mixed" khỏi hiệu ứng "ít bước hơn".
- **Kết quả (14/11, 3 seed): no-go.** Bias 0, bientail_mono vs quick_base: HardCons EN −2.4, ES −2.3, EU −1.2; F1 −2.3/−1.8/−0.8. Ở bias OOF (−8.5 vs −5) vẫn thua 1.9–2.9 HardCons. bientail_all vs m0 (bias 0): HardCons −1.0/0.0/+0.6/+0.6 (EN/ES/EU/Mixed), F1 −1.1/+0.5/0.0/−0.5; ở bias OOF (−16) F1 giảm 2–5 điểm. Bất đồng EN/ES/EU tăng (26–27% vs 23.5%). Giữ làm kết quả âm: ràng buộc nhất quán theo cấu trúc không thay được đầu 4 nhãn + self_swap.
- **Control full_1ep** (lưu ý: 4 track × 1 epoch = ít bước hơn bản mono 3 track × 3 epoch, không bằng): so với m0 (3 epoch) tốt hơn ở mọi track, HardCons +1.8/+1.3/+3.3/+2.7, F1 +0.7/+0.3/+1.0/+0.7, std nhỏ. So với quick_base: HardCons ngang (−0.3/0.0/+0.9), F1 EN thấp hơn 2.0. Kết luận: phần lớn lợi ích HardCons của bản mono đến từ việc train ít hơn (3 epoch trên 4 track là overfit vì mỗi cặp lặp lại 4 ngôn ngữ), không phải từ việc bỏ Mixed; riêng F1 EN có thể còn lợi từ mono. Việc tiếp theo: OOF + bias cho full_1ep, rồi so 1 vs 2 epoch trên train-OOF.
- **Luật chọn số epoch** (chốt 14/11 trước khi chạy): 4 track, đầu 4 nhãn, pair-aware; ứng viên 1, 2 epoch (3 epoch = m0 đã có OOF). Mỗi ứng viên: OOF 3 fold, bias chọn trên OOF (argmax trung bình 3 metric). Chọn số epoch có objective OOF tại bias của nó cao nhất. Dev chỉ dùng để báo cáo, không dùng để chọn.
- **Kết quả (14/11):** objective train-OOF tại bias của từng bản: 1 epoch 0.8615 (bias −3), **2 epoch 0.8622 (bias −8)**, 3 epoch 0.8605 (bias −10). Theo luật: **chọn 2 epoch, bias −8**. Chênh lệch 0.0007 nằm trong nhiễu, nên thực chất 1 và 2 epoch hoà nhau; dev (chỉ báo cáo) cũng hoà: trung bình 3 metric EN 0.891 vs 0.885, ES 0.865 vs 0.866, EU 0.847 vs 0.853, Mixed 0.846 vs 0.851 (2 vs 1 epoch). So với m0 (3 epoch, −10), bản 2 epoch: HardCons +2.4/+0.6/+2.1/+1.3, SoftCons +2.1/+1.7/+2.6/+2.0, F1 −0.2/−2.0/−1.9/−1.6 (EN/ES/EU/Mixed).

## 2d. Việc tiếp theo: chạy trên Kaggle (ghi ngày 14/11, làm 15/11/2026)

**Bối cảnh.** Hệ thống chính hiện tại: mDeBERTa-XNLI, pair-aware, 4 track, 2 epoch, `self_swap`, bias NEG −8 (config `m0_mdeberta_xnli_pair_all.yaml`). Dev EN (ensemble 3 seed): 0.802 / 0.971 / 0.906. Public tốt nhất không dùng artifact NEG (jhinezeal123, chỉ EN, chọn trên dev): 0.872 / 0.964 / 0.917, bằng Qwen2.5-1.5B LoRA + label smoothing 0.1 + trung bình hai chiều + nhân P(NEG) 0.8.

**Phân tích lỗi (bias −8):** hơn 50% lỗi là NEG bị đoán thành có kéo theo (recall NEG: EN 36%, ES 29%, EU 20%); 27–43% là nhầm EQ với quan hệ một chiều; lỗi dồn vào các cặp thay từ cần kiến thức từ vựng (subst:none/oov khoảng 30% sai, so với add 11%).

**Các bước, theo thứ tự:**
1. **Chốt luật chọn bias trước khi chạy** (local, khoảng 1 phút): báo cáo 2 điểm vận hành, đều chọn trên train-OOF: (a) tối ưu F1 + HardCons (giống jhinezeal123, giữ NEG); (b) tối ưu trung bình 3 metric (như hiện tại). Cần thêm tuỳ chọn objective vào `scripts/tune_neg_bias.py`.
2. **Kaggle: Qwen2.5-1.5B LoRA trên EN** (`configs/qwen_cls_en.yaml`, đã có sẵn; lần chạy local cũ chưa xong). 3 seed, decoders `[argmax, self_swap]`, lưu cả swap probs. Thêm OOF 3 fold để chọn bias. Mục tiêu: tái lập khoảng 0.86–0.87 F1 với luật safe.
3. **Kaggle: Qwen trên ES và EU** (cùng công thức, `train_tracks` 1 track mỗi lần), để đo độ yếu ở tiếng Basque.
4. **Stack Qwen + mDeBERTa** bằng một LR trên train-OOF, mỗi track một bộ trọng số (local, từ probs đã lưu). Trọng số theo ngôn ngữ dùng làm phân tích cho paper.

**Chuẩn bị trước khi lên Kaggle:**
- Commit, push working tree. Kiểm tra repo GitHub public hay private trước; working tree có lẫn thay đổi của agent khác.
- Sửa ô config trong `kaggle/runner.ipynb` (đang trỏ `m0_deberta_base_en.yaml` cũ). Settings: GPU T4×2 hoặc P100, Internet On.
- Kết quả: tải `runs.zip`, giải nén vào `runs/`, rồi chạy `scripts/overnight_report.py` (thêm nhóm `qwen_cls_en`).

**Kết quả Qwen (15/11, Kaggle 2×T4, 3 seed + OOF 3 fold mỗi ngôn ngữ).** Bias chọn trên train-OOF theo cả hai objective; dev chỉ để báo cáo. Ensemble 3 seed, `self_swap`, F1 / Soft / Hard:
- EN: bias 0: 0.866 / 0.942 / 0.903; f1_hard (−0.5): **0.870 / 0.949 / 0.910**; mean3 (−2.5): 0.850 / 0.960 / 0.917. Đạt mục tiêu bước 2 (tái lập ~0.87 F1 của jhinezeal123 mà không chọn trên dev); hơn mDeBERTa (0.802) gần 7 điểm F1 trên EN.
- ES: hai objective cùng chọn −2.5: 0.830 / 0.982 / 0.910 (bias 0: 0.834 / 0.935 / 0.874).
- EU: f1_hard (−2.0): 0.685 / 0.942 / 0.776; mean3 (−4.0): 0.645 / 0.986 / 0.798. EU kém EN ~18 điểm F1; bias âm hại F1 EU nhiều hơn.
- Bước 4 (stack) cần probs dev + train-OOF của mDeBERTa, không có trên máy Linux → chạy lại `m0_mdeberta_xnli_pair_all` trên Kaggle (`train:main:{1,2,3}`, `oof:main`).

**mDeBERTa chạy lại trên Kaggle (15/11, fp16, 24 phút):** tái lập bản local: objective train-OOF 0.8614 ở bias −6 (local bf16: 0.8622 ở −8, trong nhiễu). Dev 3 seed ở −6 (F1/Soft/Hard): EN 0.819/0.950/0.887, ES 0.782/0.958/0.863, EU 0.750/0.953/0.838, Mixed 0.745/0.952/0.822. Config vẫn để −8.

**Stack Qwen + mDeBERTa (15/11, `scripts/stack_qwen_mdeberta.py`).** Mỗi track: `w·S_qwen + (1−w)·S_mdeb + neg_bias·[NEG]`, S = score self_swap của từng model (submission-safe). (w, bias) chọn trên train-OOF theo hai objective đã chốt; thay cho LR trong kế hoạch vì grid tối ưu thẳng metric của task, cùng luật với chọn bias. Dev = ensemble 3 seed, chỉ báo cáo. Bootstrap ghép cặp theo pair_id, 1000 lần.
- **mean3** (F1/Soft/Hard): EN **0.867/0.968/0.931** (w=0.80, −2.5) vs Qwen 0.850/0.960/0.917, mDeB 0.816/0.953/0.895; ES **0.832/0.982/0.917** (w=0.75, −3.0) vs Qwen 0.830/0.982/0.910; EU **0.760/0.982/0.874** (w=0.40, −5.0) vs Qwen 0.645/0.986/0.798, mDeB 0.752/0.968/0.848.
- Stack − Qwen (mean3): EU F1 +11.5 [+6.7, +16.1], Hard +7.6 [+2.8, +12.5]; EN F1 +1.7 [−0.1, +3.7] (P>0 0.97), Hard +1.5 [0.0, +3.4]; ES hoà.
- **f1_hard:** stack không hơn Qwen ở EN (F1 −0.6, P>0 0.20), ES mất SoftCons −3.6; EU vẫn +10.6 F1.
- Trọng số theo ngôn ngữ (cho paper): EN/ES nghiêng Qwen (0.75–0.80), EU nghiêng mDeBERTa (0.40): Qwen yếu tiếng Basque, mDeBERTa (pre-train XNLI đa ngữ, train 4 track) bù.
- **Chốt:** hệ thống chính track 1–3 = stack mean3; track 4 = mDeBERTa một mình (chưa có Qwen cho Mixed).
- CodaBench dev (15/11, `dev_stack_main.zip`): khớp scorer local; macro 4 track F1/Soft/Hard 0.801/0.974/0.889 (safe cũ: 0.756/0.979/0.870). Hai lần nộp 14/11 có F1 cao hơn (0.830, 0.819) dùng pool theo pair_id, cấm ở test (2c), chỉ để phân tích.

**Thí nghiệm kế tiếp: Qwen train trên 4 track (chốt 15/11, trước khi chạy).** `configs/qwen_cls_all.yaml`: cùng công thức Qwen, train/eval 4 track, **1 epoch** (4 track ≈ 9× số dòng một track; 1 epoch ≈ 3× số bước của bản đơn ngữ 3 epoch; mDeBERTa overfit ở 3 epoch trên 4 track). Kaggle: 3 seed + OOF 3 fold.
- **Luật:** mỗi track EN/ES/EU, stack với `qwen_cls_all` thay cho Qwen đơn ngữ nếu objective mean3 trên train-OOF (tại (w, bias) tốt nhất của nó) cao hơn stack hiện tại; ngược lại giữ. Track Mixed: dùng stack (`qwen_cls_all` + mDeBERTa) nếu objective OOF cao hơn mDeBERTa một mình. Không xét stack 3 model. Dev chỉ báo cáo.
- **Kết quả (15/11, Kaggle 51 phút cho 3 seed, OOF 36 phút).** Một mình, Qwen 4 track yếu hơn Qwen đơn ngữ (self_swap bias 0, từng seed: EN ~0.83 vs 0.854, ES ~0.79 vs 0.812, EU ~0.67 vs 0.689). Objective OOF mean3 của stack: EN 0.8941 vs **0.9021** (đơn ngữ), ES 0.8913 vs **0.8977**, EU 0.8614 vs **0.8617** → **giữ Qwen đơn ngữ** cho track 1–3. Mixed: stack (`qwen_cls_all` + mDeB, w=0.50, −4.5) **0.8529** vs mDeB một mình 0.8455 → **dùng stack cho Mixed**. Dev Mixed: 0.740/0.976/0.851 (trước: mDeB ở −6, 0.744/0.965/0.836).
- **Hệ thống chính mới:** track 1–3 = stack Qwen đơn ngữ + mDeB (mean3); track 4 = stack Qwen 4 track + mDeB (mean3). Dev macro 4 track: 0.800/0.977/0.893 (`submissions/dev_stack_main_v2.zip`).
- **Pipeline test (15/11).** `configs/main_system.yaml` (sinh từ `choices.json`), `src/dico/predict.py` (nạp run đã lưu: encoder lưu nguyên, LoRA lưu adapter + head `score`; mã hoá như lúc train, chỉ view (A, B) nên không cần nhãn), `scripts/predict_final.py` (input CSV → probs → stack → zip). Kiểm tra: `--probs saved` trên dev tái tạo y hệt `dev_stack_main_v2.zip`; model tí hon trên CPU: nạp lại khớp probs lúc train (LoRA 0, encoder 3e-8). Kaggle: train lại 15 run với `KEEP_MODEL=1` (train only; train+dev để bước cuối).
- **WordNet cho lỗi thay từ (15/11, `scripts/kg_headroom.py`, train-OOF của hệ thống chính): no-go.** Luật chốt trước: đi tiếp nếu ở một track, số lỗi stack mà nhãn WordNet đúng (headroom) ≥ 5% số lỗi và > số chỗ stack đúng mà WordNet sai (risk). Kết quả headroom/risk: EN 44/374 (8.7% lỗi), ES 30/189, EU 24/59, Mixed 513/5724. Nhãn suy từ quan hệ WordNet chỉ đúng 42–45% (Mixed 25%), kém xa stack (~80%). Lý do: đa nghĩa (lấy mọi synset), ghép mọi cặp từ khác nhau, Mixed so từ khác ngôn ngữ. Lỗi dồn vào `subst:none` (EN 41% lỗi) và `subst:oov` (EU 69% lỗi: không có lemmatizer tiếng Basque, WordNet tiếng Basque thưa): đúng phần WordNet không phủ. Kết quả âm cho paper: kiến thức từ vựng cần thiết không nằm trong WordNet ở dạng dùng được.
- **Mở rộng stack (15/11, `scripts/stack_ext.py`): no-go, giữ A ở cả 4 track.** Luật chốt trước: B (+ bias EQ; chỉ EQ/NEG được có bias vì bias FWD/BWD phá nhất quán của self_swap), C (3 model: Qwen đơn ngữ + Qwen 4 track + mDeB, track 1–3), D (B+C) chỉ thay A nếu objective OOF mean3 ≥ A + 0.003 (D phải hơn cả B, C 0.003). Gain OOF so với A: EN B +0.0027, C +0.0026, D +0.0044 (không hơn B/C đủ 0.003); ES ≤ +0.0013; EU ≤ +0.0014; Mixed B +0.0001. Dev (chỉ báo cáo) xác nhận: mọi biến thể có mean3 dev thấp hơn A (EN 0.916–0.921 vs 0.922; ES 0.903–0.907 vs 0.910; EU 0.869–0.870 vs 0.872; Mixed 0.854 vs 0.856), tức gain OOF nhỏ là do chọn thêm tham số trên cùng OOF. Stack 2 model + bias NEG đã ở mức bão hoà; cải thiện tiếp phải đến từ model nền mạnh hơn.
- **Thí nghiệm: tăng dữ liệu bằng bắc cầu (chốt 15/11, trước khi chạy).** `src/dico/augment.py`: với (A,B,l1), (B,C,l2) cùng track, thêm (A,C, compose(l1,l2)); EQ∘X = X∘EQ = X (X gồm NEG), FWD∘FWD = FWD, BWD∘BWD = BWD; một bước; hai đường mâu thuẫn thì bỏ; bỏ cặp đã có và cặp trùng dev / fold giữ lại. Train đủ: EN +692, ES +640, EU +631 dòng (+21–23%); trong OOF sinh trong từng fold chỉ từ cạnh của fold train (+270–390, thận trọng). Kiểm tra quy tắc: 20/20 cặp có sẵn được suy lại đúng nhãn. Kaggle `kaggle/runner_aug.ipynb`: Qwen đơn ngữ `_aug`, 3 seed + OOF.
  - **Luật:** mỗi track EN/ES/EU, thay Qwen đơn ngữ bằng bản `_aug` trong stack 2 model nếu objective OOF mean3 của stack ≥ hiện tại + 0.003 (EN 0.9021, ES 0.8977, EU 0.8617). Dev chỉ báo cáo.
  - **Kết quả (Kaggle 15/11, `runs/stack_qwen_aug_mdeberta`): no-go cả 3 track, giữ hệ thống chính.** Stack OOF mean3 `_aug` so với hiện tại: EN 0.8977 (−0.0044), ES 0.8971 (−0.0006), EU 0.8591 (−0.0026). Qwen một mình trên OOF: EN 0.8919 (−0.003), ES 0.8828 (+0.002), EU 0.7846 (+0.002), tức không tăng thật. Dev (chỉ báo cáo) xấu hơn rõ: F1/Soft/Hard của stack EN 0.842/0.968/0.917 (hiện tại 0.867/0.968/0.931), ES 0.815/0.975/0.899 (0.832/0.982/0.917), EU 0.755/0.971/0.877 (0.760/0.982/0.874). Giải thích khả dĩ: cạnh suy ra lệch phân phối (≈78% FWD/BWD) và mang lỗi gán nhãn của chuỗi; mô hình đã học được tính bắc cầu ngầm từ dữ liệu gốc. Smoke reload trên T4 (fp16) sau bản sửa `predict.py`: max |dp| 0.0, argmax trùng 1.0.
- Đọc cho paper: một model Qwen chung cho 4 track không thay được model riêng từng ngôn ngữ (đa ngữ làm loãng ở model 1.5B), nhưng là cách duy nhất có Qwen cho Mixed, và ở đó stack thêm +1.5 HardCons.

**Thí nghiệm: debias thiên kiến trùng từ bằng product-of-experts (chốt 22/11, trước khi chạy).** Động cơ: ~40% lỗi là NEG bị đoán thành có quan hệ (cặp có nhiều từ chung, ví dụ "Two cats / A white cat"). Tham khảo: Clark et al. 2019, He et al. 2019 (PoE debiasing, HANS).
- **Bias model:** LR trên 12 đặc trưng bề mặt (`dico.baselines.features`: độ dài, độ trùng token, tập con), fit riêng từng track; xác suất trên train là **cross-fit 5 fold theo `pair_id`**; tính cho cả (A,B) và (B,A). Chỉ dùng text, không dùng cấu trúc file.
- **Train:** `poe_bias: surface`, λ = 1 cố định (không tune). Loss có giám sát của cả hai chiều dùng log_softmax(logits) + log p_bias; số hạng nhất quán (JS) giữ trên logits model chính. **Test chỉ dùng model chính** → hợp luật an toàn (2c).
- **So sánh:** `full_2ep_poe` với `full_2ep` (mDeBERTa-XNLI, 4 track, 2 epoch, pair-aware 0.2), cùng seed (1, 2, 3; OOF 3 fold seed 41), chạy local.
- **Luật go:** objective OOF mean3 (tại bias NEG chọn trên OOF của chính nó) ≥ 0.8622 + 0.003 = **0.8652**. Go → train Qwen bản PoE trên Kaggle và stack lại với cùng luật (ngưỡng +0.003 so với stack hiện tại). No-go → giữ hệ thống chính, ghi làm kết quả âm (paper: thiên kiến bề mặt không phải nguồn lỗi chính, hoặc PoE không gỡ được nó).
- **Báo cáo kèm (không dùng để chọn):** NEG recall trên OOF, dev 3 seed tại bias OOF, phân loại lỗi 4 kiểu trên dev, độ mạnh của bias model (F1 cross-fit trên train).
- **Kết quả (22/11, local, ~32 phút): no-go, giữ hệ thống chính.** Bias model: accuracy cross-fit 50.7%, weighted F1 0.458, max-prob trung bình 0.51. OOF mean3 của `full_2ep_poe` **0.8566** (bias −8) vs 0.8622 baseline (−0.0056). OOF NEG recall 24.8% → 22.7% (precision 67.6% → 71.7%), accuracy 75.3% → 74.4%. Dev 3 seed (F1/Soft/Hard): EN 0.784/0.970/0.880 (baseline 0.805/0.969/0.899), ES 0.743/0.977/0.857, EU 0.718/0.966/0.836, Mixed 0.713/0.975/0.821. Lỗi dev trung bình 3 seed (EN, EU): NEG→có quan hệ 51.7→**66.7**, 63.3→**81.0**; có quan hệ→NEG 30.7→16.7, 30.0→18.7; nhầm chiều FWD↔BWD 3.3→**10.7**, 6.0→**11.3**; EQ↔một chiều 30.7→39.3, 59.3→60.7.
- **Đọc cho paper:** PoE làm tệ đúng kiểu lỗi nó nhắm tới và làm tăng gấp 2–3 lần lỗi nhầm chiều. Ở NLI cấp cụm từ, đặc trưng bề mặt (một bên thêm từ bổ nghĩa, quan hệ tập con, hiệu độ dài) phần lớn là **tín hiệu ngữ nghĩa thật** (monotonicity: thêm modifier giao ⇒ cụ thể hơn), không phải shortcut như trùng từ ở MNLI/HANS; gỡ nó đi là mất tín hiệu hướng. Bias model chủ yếu học "ít trùng từ ⇒ NEG", nên PoE đẩy model chính ra xa NEG (NEG→quan hệ tăng, quan hệ→NEG giảm). Lỗi NEG của model không do tin vào độ trùng từ mà do không nhận ra chi tiết khác biệt (số lượng, màu, người được nói tới). Ghép với N1.

**Thí nghiệm A: cho model thấy rõ phần khác nhau (chốt 22/11, trước khi chạy).** Động cơ: PoE cho thấy tín hiệu bề mặt là thật; lỗi NEG đến từ việc không nhận ra chi tiết khác biệt. `diff_input: true`: đoạn thứ hai của input thành `B | only in first: <từ chỉ có ở A> | only in second: <từ chỉ có ở B>` (token chữ thường, giữ thứ tự, `-` nếu rỗng; chiều (B,A) tự tính lại diff). Chỉ dùng text của instance → hợp luật 2c.
- So `full_2ep_diff` với `full_2ep`: giống hệt, trừ `diff_input` và `max_len` 64 → 96 (input dài hơn; 1.4% vượt 64, 0.04% vượt 96). 3 seed + OOF 3 fold, local.
- **Luật go:** objective OOF mean3 (bias chọn trên OOF) ≥ **0.8652** (0.8622 + 0.003). Go → thêm `diff_input` cho Qwen (cần template mới) trên Kaggle và stack lại với ngưỡng +0.003. Báo cáo kèm: NEG recall OOF, 4 kiểu lỗi trên dev.
- **Đổi nơi chạy (22/11, trước khi có kết quả OOF):** local bị tràn bộ nhớ GPU (input dài hơn + dwm), chậm ~3×; người dùng chuyển sang Kaggle. Chạy lại **cả 3 seed + OOF** trên Kaggle (T4, fp16, `kaggle/runner_diff.ipynb`) để không trộn phần cứng; baseline đổi thành bản mDeBERTa chạy trên Kaggle (fp16, OOF 0.8614 ở bias −6, mục 2d) → **ngưỡng go 0.8644**. Hai seed local (bf16) đã xong chỉ giữ để tham khảo, không dùng cho luật.
- **Kết quả (Kaggle 2×T4, 23/11, 3 seed 34 phút, OOF 23 phút): no-go.** OOF mean3 **0.8517** (bias −6.5) so với ngưỡng 0.8644 (baseline Kaggle 0.8614): −0.0097, tệ hơn cả PoE. OOF tại bias 0: 0.8312 (baseline local 0.8453). Dev 3 seed (F1/Soft/Hard): EN 0.791/0.960/0.879, ES 0.766/0.975/0.858, EU 0.735/0.958/0.826, Mixed 0.729/0.959/0.821; baseline Kaggle (bias −6) EN 0.819/0.950/0.887, ES 0.782/0.958/0.863, EU 0.750/0.953/0.838, Mixed 0.745/0.952/0.822 → F1 giảm 1.5–2.8 điểm ở mọi track. Giả thuyết: (1) đoạn thứ hai không còn giống input NLI mà mDeBERTa-XNLI đã học (phân phối lệch so với pre-train); (2) model vốn đã tự so sánh được hai cụm qua attention chéo, nên diff tường minh chỉ là nhiễu thêm; (3) ở Mixed hai cụm khác ngôn ngữ nên diff gần như là toàn bộ hai cụm. Kết hợp với PoE: cả bỏ đi lẫn làm nổi tín hiệu bề mặt đều không giúp → lỗi NEG không nằm ở tầng bề mặt.

**Thí nghiệm B: LLM lớn làm chuyên gia thứ 3 trong stack (chốt 22/11, trước khi chạy).** Qwen2.5-7B-Instruct (nf4), zero-shot có định nghĩa 4 nhãn và 4 ví dụ viết tay (không lấy từ train/dev), đọc log-prob của 4 chữ cái đáp án (A–D, theo thứ tự nhãn cố định) cho cả (A,B) và (B,A). Không train → xác suất trên train là "sạch" như OOF. Track 1–3 (Mixed để sau). Script `scripts/llm_expert.py`, Kaggle `kaggle/runner_llm_expert.ipynb`.
- **Kết quả (22/11).** LLM một mình (dev, self_swap, bias 0): EN 0.656/0.820/0.643, ES 0.599/0.834/0.588, EU 0.473/0.830/0.462 (argmax tự mâu thuẫn nặng, Soft ~0.24). Kaggle 2×T4: 28 + 15 phút. Stack chạy local với Qwen từ Kaggle 15/11 và mDeB = `full_2ep` (bản Kaggle của mDeB không còn; cùng config, OOF 0.8622 vs 0.8614; stack 2 và stack 3 so trong cùng lưới nên luật không đổi). OOF mean3 stack3 vs stack2: **EN 0.9104 vs 0.9046 (+0.0057, w_llm 0.15) → go; ES 0.9020 vs 0.8949 (+0.0071, w_llm 0.15) → go; EU 0.8614 vs 0.8596 (+0.0018) → no-go.** Dev (chỉ báo cáo, bootstrap stack3 − stack2): EN F1 −1.2 [−3.0, +0.5], Soft +0.7, Hard +0.7 [−0.7, +2.3] (mean3 +0.1); ES F1 +0.4, Soft −0.7, Hard −0.3 [−2.2, +1.5] (mean3 −0.2). Dev không xác nhận gain.
  - **Lưu ý phương pháp:** LLM không train nên xác suất train của nó "thật", còn OOF của Qwen/mDeB là model fold (train 2/3 dữ liệu) nên yếu hơn model cuối → chọn trọng số trên OOF có thể **thiên vị LLM** (w_llm thực tối ưu có thể nhỏ hơn). Trước khi đưa vào hệ nộp: chạy LLM trên test (≈ 30–45 phút Kaggle) và cân nhắc kiểm tra độ nhạy w_llm (0.05–0.15) trên OOF.
- **Luật:** mỗi track, stack 3 model `w_q·S_qwen + w_m·S_mdeb + w_l·S_llm + bias·[NEG]` (lưới w theo bước 0.05 trên simplex, bias như cũ) thay stack 2 model nếu objective OOF mean3 ≥ hiện tại + 0.003 (EN 0.9021, ES 0.8977, EU 0.8617). Dev chỉ báo cáo. Kèm số LLM một mình (self_swap, bias chọn trên train) cho paper.

**Thí nghiệm: Qwen 7B cho EU (chốt 22/11, trước khi chạy; TẠM HOÃN 22/11, chưa chạy — code + notebook đã push ở `1414b73`, chạy ở lượt sau).** `configs/qwen7b_cls_eu.yaml`: Qwen2.5-7B QLoRA (base nf4, LoRA r=16 như cũ), cùng công thức với `qwen_cls_eu` (template, lr 1e-4, 3 epoch, batch 8, label smoothing 0.1); chỉ đổi kích thước model. Kaggle `kaggle/runner_qwen7b.ipynb`: 3 seed + OOF 3 fold. Động cơ: EU là track yếu nhất (stack 0.760 F1), Qwen 1.5B một mình kém mDeB ở EU (0.645 vs 0.752), nên câu hỏi là độ yếu tiếng Basque có giảm khi model lớn hơn không.
- **Luật:** thay `qwen_cls_eu` bằng `qwen7b_cls_eu` trong stack 2 model của EU nếu objective OOF mean3 của stack ≥ 0.8617 + 0.003 (cùng ngưỡng với thí nghiệm `_aug`). (w, bias) chọn lại trên OOF như cũ (`scripts/stack_qwen_mdeberta.py --qwen qwen7b_cls_{lang} --tracks 3`). Dev chỉ báo cáo. Kèm số Qwen 7B một mình so với 1.5B (OOF và dev) cho paper.
- Nếu go: cần thêm nạp lại base 4-bit trong `predict.py` để dự đoán test (chưa có).
- Không làm trong lượt này: 7B cho EN/ES, model riêng cho tiếng Basque (Latxa). Nếu 7B no-go, Latxa là ứng viên tiếp theo, chốt luật riêng.

**Ý 2: đặt cụm từ vào câu khung (context frame)** (`scripts/context_probe.py`; chốt ngày 21/11 trước khi chạy).
- Câu hỏi: model NLI được pre-train ở cấp câu có suy luận tốt hơn khi cụm từ được đặt vào một câu khung chung không? Hợp luật an toàn vì khung chỉ dùng text của chính instance.
- Khung chọn trước, không tune: `quote` (EN `The text says: "{x}".`, ES `El texto dice: "{x}".`, EU `Testuak dio: "{x}".`) và `ellipsis` (`… {x} …`). Không dùng khung tồn tại ("There is …") vì nhiều cụm là động từ hoặc cụm giới từ.
- **Giai đoạn 1 (zero-shot, không fine-tune):** mDeBERTa-XNLI, 6 log-prob hai chiều → LR fit trên train → `self_swap` trên dev, track 1–3. So `orig` với `frame` và `orig+frame` (12 feature, tương đương gated fusion ở mức đơn giản nhất). Paired bootstrap theo `pair_id`.
- **Go sang giai đoạn 2** (fine-tune mDeBERTa trên input có khung, so với `quick_base`) nếu có ít nhất một hệ (`frame` hoặc `orig+frame`, của một khung bất kỳ) đạt Δ HardCons > 0 với khoảng tin cậy 95% không chứa 0 ở ít nhất 2/3 track, và F1 không giảm quá 1 điểm ở track nào. Ngược lại thì no-go, và không thử thêm khung hay LLM sinh khung nào nữa.
- Lưu ý: có 2 khung × 2 hệ = 4 phép so sánh, nên tín hiệu sát ngưỡng cần nhìn với nghi ngờ.
- **Kết quả giai đoạn 1 (21/11): no-go.** HardCons orig (EN/ES/EU): 0.729 / 0.679 / 0.682. Khung đơn lẻ tệ hơn orig ở 5/6 ô; `quote` trên EU giảm 7.2 điểm, CI [−12.1, −2.5]. `orig+frame` có điểm ước lượng dương ở cả 6 ô (HardCons +1.4 đến +3.6, F1 +1.9 đến +5.0), nhưng CI HardCons chỉ loại được 0 ở 1 ô (`orig+ellipsis` ES, [+1.1, +7.2]). Vì khung đơn lẻ còn kém hơn orig, phần lợi của fusion nhiều khả năng đến từ việc có thêm một view nhiễu khác (hiệu ứng ensemble), không phải từ ngữ cảnh. Đây lại là mô hình zero-shot yếu (F1 khoảng 0.65, so với 0.80 khi fine-tune), nên càng khó kỳ vọng lợi ích chuyển sang được. Đóng hướng **khung cố định**. Kết quả này không bác bỏ khung có nghĩa do LLM sinh theo nội dung (xem 2b).
- **Ý 2b: khung có nghĩa do LLM sinh** (`scripts/context_llm.py`; chốt ngày 21/11 trước khi chạy). Qwen2.5-1.5B-Instruct, greedy, few-shot viết tay (không lấy ví dụ từ dev), sinh **một** khung câu có slot `[X]` cho cả hai cụm. Khung được sinh trên cặp đã sắp thứ tự (min, max), nên (A, B) và (B, A) dùng cùng khung và self_swap vẫn nhất quán. Khung hỏng (không có `[X]`) thì dùng `ellipsis`. Chỉ test EN (giai đoạn 1 zero-shot, cùng pipeline với 2a).
  - **So sánh chính:** `orig+llm` so với `orig+ellipsis` (đối chứng cho hiệu ứng "thêm một view"). Phụ: `llm` so với `orig`.
  - **Go** (sang ES/EU và fine-tune) nếu Δ HardCons của so sánh chính > 0 với CI 95% không chứa 0, và F1 không giảm quá 1 điểm. Δ > 0 nhưng CI chứa 0: thử lại **một lần** với Qwen2.5-7B-Instruct 4-bit, cùng luật. Δ ≤ 0: no-go, đóng hẳn ý 2.
  - **Kết quả lần 1 (22/11, Kaggle T4, Qwen2.5-1.5B-Instruct, EN): chưa kết luận → chạy 7B theo luật.** Khung hợp lệ 87.7% dev. F1/Soft/Hard: orig 0.656/0.939/0.729; ellipsis 0.619/0.946/0.697; llm 0.617/0.931/0.679; orig+ellipsis 0.663/0.953/0.733; **orig+llm 0.681/0.953/0.747**. So sánh chính orig+llm − orig+ellipsis: Δ Hard +1.4 [−0.4, +3.2], Δ F1 +1.8 [−0.3, +3.7]. Phụ: llm − orig Δ Hard −5.0 [−10.9, +0.4]; orig+llm − orig Δ Hard +1.8 [0.0, +4.0]. Đọc khung bằng mắt: model 1.5B hay **làm rò nội dung của một cụm vào khung** ("A fire … killed five people [X]" cho `Five dead / Much more Deaths`; "A young girl … was born yesterday [X]" cho `a young baby girl / a young girl`), gây trùng lặp kiểu "at at the station" và sinh khung mâu thuẫn với một trong hai cụm. Tức lần 1 đo chủ yếu chất lượng generator, chưa đo được ý tưởng.
  - **Lần 2 (lần cuối):** Qwen2.5-7B-Instruct 4-bit (nf4), cùng prompt, cùng few-shot, cùng luật. Không sửa prompt, không lọc khung sau khi đã xem lần 1.
  - **Kết quả lần 2 (22/11, Kaggle T4, Qwen2.5-7B-Instruct nf4, EN): no-go, đóng hẳn ý 2.** Khung hợp lệ 96.5% dev (1.5B: 87.7%). F1/Soft/Hard: llm 0.600/0.928/0.635; **orig+llm 0.686/0.957/0.747** (orig, ellipsis, orig+ellipsis như lần 1). So sánh chính orig+llm − orig+ellipsis: Δ Hard +1.4 [−1.1, +4.1], Δ F1 +2.2 [−0.2, +4.6] → CI chứa 0, lần cuối nên dừng. Phụ: llm − orig Δ Hard −9.4 [−14.7, −4.1] (xấu hơn lần 1: −5.0); orig+llm − orig Δ Hard +1.8 [−0.8, +4.7].
  - **Đọc cho paper:** khung tốt hơn (7B, ít lỗi định dạng) lại làm view khung **xấu đi** khi dùng một mình. Khung có nghĩa thêm nội dung (sự kiện, vai nghĩa), và nội dung đó tương tác với khác biệt giữa A và B: 15% khung vẫn chép nguyên một cụm vào khung ("Three men [X] were arrested" cho `Three men / Two women`), khung hợp với một cụm nhưng sai ngữ pháp với cụm kia ("They had coffee together at [X]" + "at a table"). Phần lợi khi ghép với orig (+1.4 Hard so với đối chứng, cả hai lần) ổn định nhưng nhỏ hơn độ nhiễu của dev 660 dòng. Kết luận: với NLI cấp cụm từ, ngữ cảnh do LLM sinh không giúp nhận diện quan hệ; nếu có tín hiệu thì nó nhỏ và đến dưới dạng một view phụ cho ensemble.

**Không làm:** dịch máy (đóng sau NLLB-1.3B); bientail; ép nhất quán xuyên ngôn ngữ; augment NEG (jhinezeal123 đã thử, không cải thiện).

## 3. Methods

### 3.1 Phần lõi để lấy thứ hạng cao (không có novelty, nhưng bắt buộc phải làm)

| ID | Phương pháp | Vai trò |
|---|---|---|
| M0 | Cross-encoder phân loại 4 lớp (DeBERTa-v3, mDeBERTa) | Baseline, tái hiện mức pilot |
| M1 | Entailment hai chiều: một hàm e(X,Y) = P(X⊨Y) gọi hai lần, decode bằng bảng 4 ô | SoftCons = 1.0 theo thiết kế |
| M2 | Mở rộng M1: model large, pre-train trên MNLI/XNLI, ensemble nhiều seed | Đẩy F1 và HardCons lên |
| M3 | Distill từ LLM: dùng LLM lớn gán nhãn hoặc giải thích quan hệ, rồi train encoder | Thêm tín hiệu, tăng F1 |

Ablation tối thiểu: M0 → M0 + joint decoding → M1. Bỏ reversal augmentation vì train đã có sẵn cả hai chiều.

### 3.2 Hướng novelty (chọn 1–2)

**N1. Phân tích artifact trong dữ liệu.** Rủi ro thấp; làm ngay trong tuần 1.
Câu hỏi: model có học shortcut kiểu "cụm dài hơn thì entail cụm ngắn hơn" (*a black dog* ⇒ *a dog*) không?
- Baseline chỉ dùng heuristic: hiệu độ dài, tỉ lệ token trùng nhau, A ⊂ B.
- Bộ test đối chứng (contrast set) nhỏ: cụm dài hơn nhưng không entail, ví dụ *a dog without a leash* với *a dog on a leash*.

Nếu heuristic đã đạt F1 cao thì đó là phát hiện đáng giá: SoftCons/HardCons đang đo cả shortcut chứ không chỉ khả năng suy luận. Đây cũng là góp ý trực tiếp cho thiết kế benchmark của BTC.

**N2. Nhất quán đa góc nhìn xuyên ngôn ngữ (multi-view consistency).** Đây là hướng novelty chính được đề xuất.
Mỗi cặp gốc có 3 ngôn ngữ đơn ngữ, 6 tổ hợp trộn và 2 chiều, tức là nhiều "góc nhìn" của cùng một quan hệ.
- **Train:** thêm loss bắt model đồng thuận trên mọi góc nhìn của cùng một cặp gốc. Đây là tổng quát hoá của reversal consistency, từ 2 chiều lên 2 chiều × nhiều ngôn ngữ.
- **Test:** gom các instance thuộc cùng một cặp gốc (bằng dịch máy hoặc độ tương đồng embedding) rồi decode chung. Cần **hỏi BTC xem decode chung trên test có hợp lệ không**.

Hướng này tận dụng cấu trúc riêng của bộ dữ liệu, nên các đội khác ít khả năng làm trùng.
Kỳ vọng: ES và EU tăng điểm nhờ mượn tín hiệu từ EN.

**Related work của N2** (tra ngày 9/11/2026; chưa đầy đủ, xem ghi chú cuối mục). Từng thành phần của N2 **đều đã có người làm riêng lẻ**:

| Thành phần của N2 | Đã có | Tài liệu |
|---|---|---|
| Consistency loss xuyên ngôn ngữ khi train (KL giữa bản gốc và bản dịch) | Có, nhiều | xTune: [Zheng et al. 2021](https://consensus.app/papers/details/1b0dd6b373c85d9688e5e568a9e5bb95/); XNLI prompt consistency: [Qi et al. 2022](https://consensus.app/papers/details/bd6c14e300c0562c85502a391ba830fa/); [SoftMV, Li et al. 2023](https://consensus.app/papers/details/ea0f8ce63e125d198c727a41f71b3031/); [BCLF 2026](https://consensus.app/papers/details/2ec4067689275003bcfbd71358cc2124/) |
| Ensemble dự đoán trên nhiều bản dịch lúc test | Có | [Ponti et al. 2021, latent translation](https://arxiv.org/pdf/2107.11353.pdf); [NAACL 2024](https://aclanthology.org/2024.naacl-long.298.pdf); [Artetxe et al. 2023](https://aitopics.org/doc/arxivorg:E23B2EB3) |
| Nhất quán khi đảo premise và hypothesis | Có | [Wang et al. AAAI 2019](https://ar5iv.labs.arxiv.org/html/1809.02719); [Minervini & Riedel 2018](https://consensus.app/papers/details/1ca6dd84a70757a3958247531779ca6d/) |
| Decode chung có ràng buộc logic giữa các dự đoán | Có | [ConCoRD, Mitchell et al. 2022](https://www.arxiv.org/pdf/2211.11875) |
| Đánh giá NLI đa ngữ và code-switch | Có (chỉ đánh giá) | [Abdaljalil et al. 2025](https://ar5iv.labs.arxiv.org/html/2508.14735) |
| **Kết hợp: ràng buộc theo chiều (đảo nhãn FWD ↔ BWD) và ràng buộc xuyên ngôn ngữ (giữ nguyên nhãn) trong cùng một nhóm, có cả cặp trộn ngôn ngữ** | **Chưa tìm thấy** | — |

**Đánh giá:** novelty của N2 là **novelty kết hợp**, ở mức đủ cho một paper SemEval nhưng chưa đủ cho main conference. Để mạnh hơn, nên đóng khung N2 thành một **nhóm ràng buộc (equivariance)**:
- phép đảo chiều tác động lên nhãn bằng Rev;
- phép dịch giữ nguyên nhãn.

Đi kèm là một phân tích chưa ai làm: **khi nào dịch thuật làm đổi quan hệ entailment**. Ví dụ hình thái học của tiếng Basque, hoặc mạo từ và số ít/số nhiều bị mất khi dịch, có thể làm đổi chiều entailment. Mỗi trường hợp như vậy là chỗ hai loại ràng buộc mâu thuẫn nhau.

Ghi chú: một lượt search bị rate limit và Scholar Gateway không truy cập được, nên bảng trên chưa đầy đủ. Cần kiểm tra thêm trên Google Scholar với các từ khoá: *"cross-lingual" "directional" entailment consistency*, *multilingual NLI symmetry*, và SemEval 2027 Task 2 (các system paper của đội khác sẽ ra vào tháng 2/2027).

**N3. Order/box embeddings.** Rủi ro cao; chỉ làm nếu còn thời gian.
Biểu diễn mỗi cụm từ là một box. Khi đó A ⇒ B tương ứng với box(A) ⊆ box(B). Nhờ cấu trúc hình học này, tính bất đối xứng và tính nhất quán có sẵn. Có thể kết hợp với cross-encoder.
Rủi ro: bi-encoder thường kém cross-encoder về độ chính xác.

## 4. Thí nghiệm (điền dần; dev, 3 seed, mean ± std)

| Hệ thống | F1 | SoftCons | HardCons |
|---|---|---|---|
| Random / majority |  |  |  |
| N1: chỉ dùng heuristic |  |  |  |
| M0 |  |  |  |
| M0 + augmentation |  |  |  |
| M0 + joint decoding |  |  |  |
| M1 |  |  |  |
| M2 (large + NLI + ensemble) |  |  |  |
| M2 + N2 |  |  |  |

Bảng đa ngữ (giá trị là F1 / HardCons):

| Hệ thống | EN | ES | EU | Mixed |
|---|---|---|---|---|
| mDeBERTa M1 |  |  |  |  |
| + N2 consistency |  |  |  |  |
| Translate-test |  |  |  |  |

## 5. Hạ tầng

- **Local (RTX 5060, 8 GB):** dùng làm máy chính, cho các model base (DeBERTa-v3-base, mDeBERTa-base) và suy luận LLM 7B dạng 4-bit. Cài WSL2, PyTorch cu128 trở lên, train với `bf16=True`.
- **Kaggle hoặc Colab:** dùng cho DeBERTa-v3-large và XLM-R-large.
- Code đặt trên git; cấu hình bằng config YAML; mỗi lần chạy ghi kết quả vào `results/*.csv`.

## 6. Timeline và checklist

Evaluation diễn ra từ 10/1 đến 31/1/2027; hạn nộp paper là tháng 2/2027 (chưa có ngày cụ thể).

**P0. Chuẩn bị (9–15/11)**
- [ ] Đăng ký trên CodaBench; clone repo task; đọc paper pilot (LREC 2026) và paper PhrasIS
- [ ] Cài môi trường; kiểm tra GPU
- [ ] Chạy scorer trên baseline random/majority
- [ ] **Hỏi BTC:** có được dùng các instance test khác để decode chung không (ảnh hưởng tới N2)?

**P1. Dữ liệu, N1 và M0 (16–29/11)**
- [ ] EDA; kiểm tra train đã có sẵn cả hai chiều chưa
- [ ] N1: baseline heuristic (độ dài, độ trùng token)
- [ ] M0 đạt F1 ≈ 0.75 trên dev EN; nộp thử lên CodaBench

**P2. Phần lõi M1–M2 (30/11–20/12)**
- [ ] M1 entailment hai chiều; ablation M0 → M1
- [ ] M2: model large, pre-train MNLI/XNLI, 3 seed

**P3. Đa ngữ và N2 (21/12–3/1)**
- [ ] mDeBERTa và XLM-R trên ES, EU, Mixed
- [ ] N2: loss đa góc nhìn; decode theo nhóm (nếu BTC cho phép)

**P4. Chốt hệ thống (4–9/1)**
- [ ] Chọn hệ thống và ensemble cho từng track; viết script dự đoán chỉ bằng một lệnh

**P5. Evaluation (10–31/1)**
- [ ] Nộp cả 4 track kèm metadata; lưu lại mọi file nộp

**P6. Paper (tháng 2)**
- [ ] Viết paper; nhờ người đọc góp ý 1 vòng; public code; nộp bài

## 7. Dàn ý paper (khoảng 5 trang, template ACL)

Intro (RQ và đóng góp) · Background · System (M1, N2) · Setup · Results · Analysis (N1, error analysis) · Conclusion · Limitations

## 8. Tài liệu tham khảo (kiểm tra lại trước khi trích dẫn)

- Paper pilot LREC 2026: https://lrec.elra.info/lrec2026-main-423
- PhrasIS: https://doi.org/10.1093/jigpal/jzae037
- MacCartney & Manning: natural logic (nguồn gốc của các quan hệ forward/reverse entailment và equivalence)
- Minervini & Riedel (2018): adversarial regularisation cho NLI consistency
- Li et al. (2019): logic-driven consistency cho mô hình neural
- Kuhn et al. (2023): semantic entropy (dùng bidirectional entailment để xác định tương đương)
- Vendrov et al. (2016): order embeddings; Vilnis et al. (2018): box embeddings
- Gururangan et al. (2018): annotation artifacts trong NLI (cơ sở cho N1)
