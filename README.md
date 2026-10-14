# NLD fused speculative decoding

## 설치

Python 3.11, CUDA 12.4

```
python3.11 -m venv venv
source venv/bin/activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## 실행

```
bash scripts/run.sh gsm8k runs/gsm8k
```

MATH-500은 `gsm8k` 대신 `math500`을 넣습니다. 결과 표는 `runs/gsm8k/table.md`에 저장됩니다.

## 파일

| 파일 | 역할 |
|---|---|
| `scripts/run.sh` | 아래 단계를 순서대로 실행 (비용표 → 시간·정확도 측정 → FLOPs 기록 → 결과 표) |
| `scripts/slurm_example.sbatch` | SLURM 작업 예시 |
| `cost_table.py` | 실행하는 GPU에서 forward 시간을 재서 블록별 선택에 쓸 비용표 생성 |
| `bench.py` | baseline과 최종 방법의 시간·정확도 측정 |
| `flops_record.py` | 시간 측정 없이 다시 생성하면서 FLOPs 기록 |
| `analyze.py` | 결과 표 (`table.md`) 생성 |
| `spec_fused.py` | 최종 방법의 생성 루프 |
| `spec_trace.py` | baseline 생성 루프 복사본 (FLOPs 기록용) |
| `flops.py` | FLOPs 계산식 |
| `scoring.py` | 채점 (GSM8K: 저자 SGLang 채점기, MATH-500: math_verify) |
| `common.py` | 모델·데이터 로딩 등 공통 함수 |
| `checks.py` | 구현 검사 (마스크 구조, 정보 누출, 생성 불변식). `run.sh`에는 없고 `python checks.py`로 따로 실행 |
