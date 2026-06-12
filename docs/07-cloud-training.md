# 07 · 클라우드 학습 런북 (RunPod)

> 용도: 검출 사다리 3-seed 매트릭스, 증강 ablation, P2b 속도 head 등 GPU 학습 가속.
> **원본 오디오(28 GB+)는 불필요** — 멜 캐시가 자급자족: 인덱스는 `cache/sources.json`,
> 멜은 `cache/mel/*.npy`만 읽는다 (캐시 완비 시 wav 접근 0회). 보낼 것 = 코드 + `cache/` = 3.1 GB.

## 0. 보낼 것 만들기 (로컬 맥)

```bash
cd ~/PycharmProjects && tar -cf /tmp/airacle_pod.tar \
  Airacle/cache Airacle/results Airacle/*.py Airacle/requirements.txt
```

## 1. Pod 생성 (RunPod 콘솔)

- **Pods → Deploy**: GPU는 **RTX 4090** 권장 (이 워크로드는 모델이 작아 4090이면 충분,
  community ~$0.3–0.4/h). vCPU 많은 인스턴스 선호 (DataLoader 워커용, ≥8).
- **Template**: 공식 `runpod/pytorch` (CUDA 12.x). Container Disk **20 GB+**.
- 연결: 콘솔의 **Connect → Web Terminal** 또는 SSH (Connect에 ssh 명령이 표시됨).

## 2. 파일 전송 (둘 중 하나)

**A. runpodctl (간단 — 키 설정 불필요)**
```bash
# 맥:  brew install runpodctl  (최초 1회)
runpodctl send /tmp/airacle_pod.tar          # → 코드(번호) 출력
# Pod 터미널:
runpodctl receive <코드>
```

**B. scp** (콘솔 Connect의 SSH 정보 사용)
```bash
scp -P <포트> /tmp/airacle_pod.tar root@<ip>:/workspace/
```

## 3. Pod에서 실행

```bash
cd /workspace && tar -xf airacle_pod.tar && cd Airacle
pip install numpy scipy                       # torch는 이미지에 포함
python3 train.py --model cnn --limit 500 --epochs 1 --workers 8   # 스모크 (~1분, cuda 확인)

# 본 실행 — seed 43, 44 (run_ladder는 results 기반 멱등이라 중단·재실행 안전)
nohup bash -c 'python3 run_ladder.py --seed 43 && python3 run_ladder.py --seed 44' \
  > /workspace/ladder.log 2>&1 &
tail -f /workspace/ladder.log                 # 진행 확인
```

run_ladder가 모델×증강 9조합을 seed별로 순차 실행하고 `results/ladder.json`에 누적한다.
(맥에서 seed 42가 돌고 있으면 seed만 나눠서 충돌 없음 — 키가 (model, aug, seed).)

## 4. 결과 회수 + 종료

```bash
# 맥에서:
scp -P <포트> root@<ip>:/workspace/Airacle/results/ladder.json /tmp/ladder_pod.json
scp -P <포트> -r root@<ip>:/workspace/Airacle/models /tmp/models_pod   # 체크포인트 (선택)
```

회수한 `ladder.json`은 로컬 것과 **병합**한다 (둘 다 리스트 — concat 후 저장; (model, aug, seed) 중복만 주의).

**⚠ 다 끝나면 Pod를 Stop이 아니라 Terminate** — Stop 상태도 디스크 과금이 계속된다.

## 비용 감각

- 사다리 9런 ≈ 4090에서 2–4 h/seed → seed 2개 ≈ **$2–4**.
- 장시간 안 쓰면 무조건 Terminate. 데이터는 로컬 캐시에서 언제든 재타르.

## 주의

- `cache/sources.json`의 wav 경로는 맥 절대경로지만 **문자열로만** 사용됨
  (멜 캐시 키 = 상대경로 sha1, split 검증 = 경로 부분문자열) → Pod에서 수정 불필요.
- 캐시가 불완전하면 `mel_for_file`이 wav를 읽으려다 실패한다 — 보내기 전 `ls cache/mel | wc -l`
  = **28,643** 확인.
- Pod에는 caffeinate·nohup 분리가 필요 없다 (서버는 안 잔다). 단 웹터미널 세션이 끊겨도
  nohup이면 계속 돈다.
