# 데이터와 선행연구 확인 기록

확인 기준일: 2026-10-23. 아래 자료는 연구 설계·데이터 선택을 위해 직접 확인했다.
선행연구의 수치나 예측 파일을 본 프로젝트 실험 결과로 사용하지 않는다.

## 실제로 사용한 데이터

1. Correig-Fraga, E., Guimerà, R., Sales-Pardo, M. **Data and source data for
   “Structure alone supports efficient visual computation in the Drosophila visual system”**,
   v1.0.0. [Zenodo 21549559](https://zenodo.org/records/21549559), CC BY 4.0.
   이 연구의 저자가 공개한 실제 FlyWire 연결망의 가공본, 뉴런 주석, 오른쪽 망막 좌표를 사용한다.
   저자들이 제공한 무작위 연결망과 과제 예측 결과는 다운로드하거나 성능 비교에 재사용하지 않는다.
   본 프로젝트의 재배선과 자극은 독립적으로 생성한다.
2. FlyWire Consortium. **FlyWire Whole-brain Connectome Connectivity Data**, v783.0.
   [Zenodo 10676866](https://zenodo.org/records/10676866).
   위 가공 연결망의 상위 원자료이다. 추가 검증 단계에서 852MB feather 원본을 다운로드하고
   기탁 MD5를 확인했다. 모델의 모든 연결·시냅스 수·해부학적 이웃 선택이 정확히 일치했다.
3. Schlegel et al. **FlyWire annotations v2.1.0**.
   [고정 릴리스](https://github.com/flyconnectome/flywire_annotations/releases/tag/v2.1.0).
   가공 데이터 기탁에 포함된 동일 버전의 주석 사본을 사용한다.

다운로드 파일별 SHA-256은 기탁의 SHA256SUMS와 비교한다.
`data/raw/zenodo_21549559/download_manifest.json`과
`data/processed/flywire_visual.json`에 검증값과 출처를 기록한다.

## 기획안에 포함된 연구

- Cho, J., Kang, M., Kim, D. (2026). **A Drosophila Whole-Connectome Network Can Learn
  Human-Designed Cognitive Tasks**. [arXiv:2610.10014](https://arxiv.org/abs/2610.10014).
  연결 위치를 고정하면서 해부학적 edge의 스칼라 가중치를 학습한다.
  본 프로젝트는 내부 가중치까지 고정하고 단일 수량 출력층의 순차 전이를 평가한다.
- Yu et al. (2025). **Biological Processing Units**.
  [arXiv:2507.10951v1](https://arxiv.org/html/2507.10951v1).
  고정된 유충 연결망에 학습 가능한 입력·출력부를 결합한다.
- Therianos (2026). **A frozen rate operator from the complete larval connectome**.
  [arXiv:2606.17745v2](https://arxiv.org/abs/2606.17745).
  고정 연결망의 전체 동역학과 정확한 배선에 의존하는 입력 전달을 구분한다.
- **Topological Sensitivity in Connectome-Constrained Neural Networks** (2026).
  [arXiv:2604.04033](https://arxiv.org/abs/2604.04033).
  대조군과 설정 선택의 공정성을 고려해야 할 근거로 검토한다.
- Bengochea et al. (2023). **Numerical discrimination in Drosophila melanogaster**.
  Cell Reports 42, 112772. [PubMed](https://pubmed.ncbi.nlm.nih.gov/37453418/),
  [논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC10442639/).
  실제 초파리의 수량 구별과 LC11 억제 실험이다.
- **Object displacement-sensitive visual neurons drive freezing in Drosophila**.
  [논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC8716191/).
  LC11의 일반적인 작은 물체 처리 기능을 함께 고려해야 할 근거이다.

## 추가로 확인한 가까운 선행연구

Correig-Fraga et al. (2026). **Structure alone supports efficient visual computation in the
Drosophila visual system**. [arXiv:2610.10023v1](https://arxiv.org/html/2610.10023v1).
실제 망막 형태를 이용한 시각 입력과 연결망 기반 수량 구별을 다룬다.
따라서 “실제 연결망으로 수량을 판독한다”는 사실만으로 새 기여를 주장하면 안 된다.
해당 모델에는 학습 가능한 시냅스 gain·뉴런 threshold 조건이 있으며,
본 프로젝트는 고정 입력·고정 내부 가중치·선형 판독기 및 **수량→순차 합산 전이**에 초점을 맞춘다.
완전한 독창성 판단은 추가 체계적 문헌 검토가 필요하다.

## 대안 자료 접근 확인

[MaleCNS 다운로드](https://male-cns.janelia.org/download/)에는 bulk 자료와
neuPrint API 접근 방법이 있다. 이번 실행은 망막 좌표와 연결망을 같은 버전으로
확보할 수 있는 FlyWire 가공 기탁을 선택했다.
[FlyWire Codex FAQ](https://codex.flywire.ai/faq)는 대량 분석에 정적 다운로드를 권한다.
로그인 필요 페이지의 접근 제한을 우회하지 않았다.
