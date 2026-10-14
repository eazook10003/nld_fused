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
