# models/ — 체크포인트 매니페스트

**어떤 모델을 쓰나?** → 아래 "현역" 표만 보면 된다. 나머지는 전부 ablation·구세대 아카이브다.
`.trt` 엔진은 레포에 없다 — 젯슨에서 [`../jetson_build.sh`](../jetson_build.sh)로 ONNX에서 빌드하고, [`../run.sh`](../run.sh)가 파일 존재를 감지해 자동 선택한다.

## 현역 (배포 — run.sh 자동 선택)

| 역할 | 파일 | 근거 |
|---|---|---|
| 검출 확정 (5 s 창) | `cnn_attn_full_s42` (.pt/.onnx) | P1 사다리: 0 dB recall 0.91, ViT 동급·7.6× 작음 (`results/ladder.json`) |
| 검출 예비 (1.5 s 창) | `cnn_attn_full_s42_65f.onnx` | 확정과 **같은 가중치**, 입력 창만 65프레임 — PRE 실측 1.76 s (`0afcdcc`) |
| 검출 예비 (2 s, 폴백) | `cnn_attn_full_s42_87f.onnx` | 65f 엔진 없을 때 run.sh가 대신 선택 (PRE ≈2.7 s) |
| 속도+방향 | `speed_neural_dir` (.pt/.onnx) | 방향 3클래스 헤드(정지/접근/멀어짐) (`b51a629`) — **⚠ 실주행 미검증(잠정)** |
| 차종 | `subtype_cnn_attn_yt_s42` (.pt/.onnx) | 유튜브 실채널 파인튜닝 — held-out 1/5→3/5, in-domain 89→86 트레이드 (`c8e5968`, `40957d8`) |

함정 주의: 차종은 3세대가 공존한다(`_s42`→`_dom`→`_yt`). in-domain 정확도는 `_dom`(89%)이 제일 높지만 **실채널에서 무너져서**(1/5, '전부 구급차' 붕괴) 배포는 `_yt`다. 숫자만 보고 고르지 말 것.

## 아카이브 (배포 금지 — 재현·비교용)

| 파일 | 무엇 |
|---|---|
| `cnn_none_s42`, `cnn_attn_{none,wave,wave_spec}_s42`, `vit_{none,wave,wave_spec,full}_s42` | P1 검출 사다리 증강 ablation — 결과는 `results/ladder.json` |
| `speed_neural` (.pt/.onnx) | 방향 헤드 없는 구세대 속도망 — `_dir` 없으면 run.sh가 폴백 |
| `speed_neural_dom.pt` | 속도 채널증강 실험 (`results/speed_dom_train.log`) |
| `subtype_cnn_attn_s42`, `subtype_cnn_attn_dom_s42` | 차종 구세대(무증강/채널증강) — `_yt`로 대체. `_dom`은 `_yt` 파인튜닝의 base |
| `subtype_cnn_s42.pt` | 차종 PlainCNN 대조군 |

`*.onnx.data`는 ONNX 외부 가중치 파일 — 짝이 되는 `.onnx`와 함께 옮겨야 한다.

## 파일명 규약

`{모델}_{증강}_s{시드}[_{창프레임}f]` — 예: `cnn_attn_full_s42_65f` = CNNAttn·full 증강·시드 42·65프레임(1.5 s) 창.
차종은 `subtype_`, 속도는 `speed_` 접두. `_dom`=채널(도메인)증강, `_yt`=유튜브 실채널 파인튜닝, `_dir`=방향 헤드 포함.
