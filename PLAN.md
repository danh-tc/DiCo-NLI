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
