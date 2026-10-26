# Frozen Fruit Fly Connectome: 수량에서 합산으로

사용자 제공 기획안을 실행 가능한 연구 코드와 재현 자료로 옮긴 프로젝트입니다.
주 실험에서는 내부 연결·가중치·시각 입력을 고정하고 선형 출력층만 학습합니다.
단일 점 집합에서 학습한 출력층을 두 집합의 순차 제시에도 그대로 적용합니다.
현재 진행 범위와 핵심 결과는 [`PROJECT_STATUS.md`](PROJECT_STATUS.md)에 있습니다.

## 결과 보기

실제 데이터 예비실험: [`results/pilot/REPORT.md`](results/pilot/REPORT.md)
또는 [`results/pilot/report.html`](results/pilot/report.html).
세부 해석과 실패 양상: [`results/pilot/INTERPRETATION.md`](results/pilot/INTERPRETATION.md).
`results/smoke`는 **합성 그래프를 사용한 코드 검증**이며 초파리에 대한 실험 결과가 아닙니다.

## 설치와 실행 (Windows PowerShell)

Python 3.11 이상과 uv가 필요합니다. GPU는 필요하지 않습니다.
명령은 이 README가 있는 폴더에서 실행합니다.

```powershell
uv --cache-dir .uv-cache venv .venv
uv --cache-dir .uv-cache pip install --python .venv\Scripts\python.exe -r requirements-lock.txt
uv --cache-dir .uv-cache pip install --python .venv\Scripts\python.exe --no-deps --no-build-isolation -e .

# 실제 데이터 약 188 MB, 출처 및 SHA-256 검증 포함
python scripts\fetch_data.py
.venv\Scripts\python.exe -m flynum prepare

# CPU 과다 스레드 생성을 제한하여 작은 행렬 계산의 낭비를 줄임
$env:OPENBLAS_NUM_THREADS = '2'
$env:OMP_NUM_THREADS = '2'
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m flynum run --config configs\pilot.json --out results\my-pilot --include-c
```

한 번에 실행하려면 `powershell -ExecutionPolicy Bypass -File scripts\run_project.ps1 -Mode pilot -IncludeC`를 사용합니다.
스크립트는 작업 폴더 안에 환경·캐시·자료·결과를 저장합니다.
네트워크가 제한된 실행 환경에서는 설치와 데이터 다운로드에 네트워크 접근 허용이 필요합니다.

빠른 합성 데이터 실행:

```powershell
.venv\Scripts\python.exe -m flynum demo --config configs\smoke.json --out results\my-smoke --include-c
```

이미 완료된 모델은 같은 코드·설정·그래프일 때 `--resume`으로 재사용할 수 있습니다.
코드나 설정이 바뀌면 새로운 결과 폴더를 사용합니다.
보고서만 다시 생성하려면 `.venv\Scripts\python.exe -m flynum report results\my-pilot`을 실행합니다.

저장한 결과의 독립 검증:

```powershell
.venv\Scripts\python.exe scripts\verify_run.py results\my-pilot
.venv\Scripts\python.exe scripts\analyze_failures.py results\my-pilot
```

상위 FlyWire Feather 원본(추가 약 852MB)과 실제 부분 회로를 대조하는 도구도 포함했습니다.
`python scripts\fetch_canonical.py`로 기탁 MD5를 검증한 원본을 확보한 뒤
`.venv\Scripts\python.exe scripts\audit_canonical.py`로 모든 모델 연결·시냅스 수·선택 규칙을 검사합니다.
이 원자료 검증은 예비실험의 예측이나 설정을 변경하지 않습니다.

## 구현 범위

| 항목 | 구현 |
|---|---|
| 실제 연결망 | FlyWire FAFB v783 가공 그래프, 해부학적 기준으로 부분 회로 선택 |
| 입력 | 고정 R7 Voronoi 망막 대응, 회색조 점 자극 |
| A | 수량 회귀·반올림 정확도·별도 대소 비교 판독기·수량 비율 분석 |
| B | 추가 학습 없는 합산 전이, 순서 교환·간격·제시 시간·0 피연산자 분석 |
| 대조 | 방향성/부호 연결 정도 보존 재배선 20개, 무작위 그래프, 가중치 섞기, 재귀 제거, 영상 단서, 시간 적분기 |
| 부호 민감도 | glutamate 억제 가정과 흥분 가정을 별도 비교 |
| C | 별도 합산 출력층, 8/24/64개 학습 예제, 교환쌍 그룹 분할, 미학습 정답·경계 합 분리 |
| D | 실행 조건 검토 문서화. 독립적인 입력 대응 검증·일반 시각 과제 확보 후 확장 |
| 재현 | 자극 이미지·메타데이터·원시 예측·판독기·설정·환경·코드/자료 해시 |

모든 모델의 ridge 탐색 예산은 동일합니다. A validation에서만 alpha를 선택하고,
B에서 출력층 해시가 유지되는지 확인합니다. 공통 alpha=1 비교도 저장합니다.

## 과학적 범위

현재 실행은 탐색적 예비실험입니다. 외부 사전등록 또는 독립 확인 실험을 대신하지 않습니다.
시냅스 개수, transmitter 기반 부호, 추정한 망막 평면 좌표, 단순 tanh 동역학을 사용합니다.
실제 초파리가 이 모델과 같은 방식으로 수량이나 덧셈을 처리한다는 결론을 내리지 않습니다.
낮은 성능과 실제 연결망의 우위가 없는 결과도 그대로 기록합니다.

추출 규칙과 통계 해석은 [`docs/protocol.md`](docs/protocol.md),
출처와 선행연구는 [`docs/sources.md`](docs/sources.md),
후속 작업은 [`docs/next_steps.md`](docs/next_steps.md)에 있습니다.
원본 기획안은 [`docs/proposal_original.md`](docs/proposal_original.md)에 보관했습니다.

## 파일 구조

```text
configs/          예비실험 및 빠른 검증 설정
src/flynum/       자극, 연결망, 준비, 고정 모델, 실험, 보고서, CLI
scripts/          공개 데이터 다운로드 및 PowerShell 실행 도구
tests/            데이터 누출·방향·기억·재배선·부호·판독기 고정 검사
data/raw/         공개 기탁 원본과 체크섬 (다운로드 생성)
data/processed/   추출 연결망·데이터 명세·전체 뉴런 선택/제외 목록
results/pilot/    실제 연결망 예비실험과 대조군 결과
results/smoke/    합성 그래프 코드 검증 결과
docs/             프로토콜·출처·후속 연구 계획
```

데이터는 원저자들의 CC BY 4.0 자료입니다. 데이터 사용 시 가공 자료와 원자료,
FlyWire 주석 자료를 모두 인용해야 합니다. 구체적 출처는 `docs/sources.md`에 있습니다.
`deliverables/frozen-fly-project.zip`에는 코드·문서·처리된 부분 회로·실험 결과가 포함됩니다.
가상환경·캐시·대용량 공개 원본은 제외하며, 원본은 다운로드 스크립트로 다시 확보할 수 있습니다.
