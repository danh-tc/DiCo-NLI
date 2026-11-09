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
