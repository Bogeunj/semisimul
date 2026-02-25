# proc2d 프로젝트 현황 보고서

작성일: 2026-02-25  
대상 경로: `/home/bogeun/projects/semisimul`

## 1) 문서 목적

이 문서는 `proc2d` 저장소의 현재 구현 범위, 검증 결과, 작업트리 상태를 한 번에 확인하기 위한 최신 리포트다.

---

## 2) 요약 (Executive Summary)

현재 기준 결론:

- Core 파이프라인(`mask -> oxidation(optional) -> implant -> anneal -> analyze -> export`) 동작
- P2 확장 기능(산화/Arrhenius/material map/tox 출력) 반영 상태
- CLI/GUI 모두 사용 가능
- 테스트 통과: **12 passed**
- 예제 deck 실행 확인:
  - `deck_analysis_history_vtk`: exports=12
  - `deck_oxidation_implant_anneal`: exports=12
  - `deck_schedule_anneal`: exports=2
- Git 작업트리는 clean 아님
  - 수정: `scripts/setup_venv.sh`
  - 신규(추적 안됨): `The`, `This` (빈 파일)

---

## 3) 프로젝트 개요

- 프로젝트 성격: 반도체 공정 2D 단면(Process 2D cross-section) 시뮬레이터
- 패키지명/버전: `proc2d / 0.1.0`
- 언어/런타임: Python 3.10+
- 빌드/배포: `pyproject.toml` + setuptools editable install
- 실행 인터페이스:
  - CLI: `python3 -m proc2d run <deck.yaml> --out <dir>`
  - GUI: `python3 -m streamlit run proc2d/gui_script.py --server.port 8502`

---

## 4) 저장소/형상관리 상태

### 4.1 Git 상태

- 브랜치: `main` (원격 `origin/main` 트래킹)
- 원격: `https://github.com/Bogeunj/semisimul.git`
- 현재 작업트리:
  - `M scripts/setup_venv.sh`
  - `?? The`
  - `?? This`

### 4.2 디렉터리 구성 요약

- 소스: `proc2d/`
- 테스트: `tests/`
- 예제 deck:
  - `examples/deck_basic.yaml`
  - `examples/deck_analysis_history_vtk.yaml`
  - `examples/deck_oxidation_implant_anneal.yaml`
  - `examples/deck_schedule_anneal.yaml`

---

## 5) 기능 구현 현황

### 5.1 시뮬레이션 코어

- 1D mask openings + lateral Gaussian smoothing
- 2D separable implant (`tox(x)` 반영)
- oxidation step (Deal-Grove, 표면 이동, Si/SiO2 material map)
- implicit diffusion anneal
  - 상수 `D_cm2_s`
  - Arrhenius(`D0`, `Ea`, `T`) + schedule
- mixed top BC
  - open: Robin/Neumann/Dirichlet
  - blocked: Neumann
- oxide barrier 제어 (`oxide.D_scale`, `cap_eps_um`)

### 5.2 분석/출력

- Analyze step
  - `metrics.json`, `metrics.csv`, `sheet_dose_vs_x.csv`
  - total mass, peak, junction/lateral 지표 계산
- Anneal history
  - `history.csv`, `history.png`
  - 필드: `time_s`, `mass`, `peak_cm3`, `flux_out`, `residual`
- Export
  - 기본: `npy`, `csv`, `png`
  - 시각화: `vtk` (`C.vtk`, `C_log10.vtk`, `material.vtk`)
  - 산화막: `tox_vs_x.csv`, `tox_vs_x.png`

### 5.3 GUI

- 실행/파라미터 조정 + 결과 탭 제공
  - `Map`, `Linecuts`, `Metrics`, `History`, `Compare`, `Artifacts`
- 다운로드/보조 기능
  - VTK, ZIP 다운로드
  - 최근 2회 before/after 비교
  - full C 세션 저장(옵션)

---

## 6) 테스트/검증 현황 (2026-02-25 기준)

### 6.1 테스트 실행

- 명령: `.venv/bin/python -m pytest`
- 결과: `12 passed in 0.88s`

테스트 커버(파일):

- `tests/test_mass_conservation.py`
- `tests/test_symmetry.py`
- `tests/test_1d_limit.py`
- `tests/test_metrics.py`
- `tests/test_history.py`
- `tests/test_vtk_writer.py`
- `tests/test_oxidation_p2.py`

### 6.2 예제 deck 실행

1) Analyze/History/VTK deck

- 명령: `.venv/bin/python -m proc2d run examples/deck_analysis_history_vtk.yaml --out outputs/run2`
- 결과: `Done. Grid=(201, 401), exports=12`
- 생성물: `C*.vtk`, `history.*`, `metrics.*`, linecut CSV, `sheet_dose_vs_x.csv`

2) P2 산화 + Arrhenius deck

- 명령: `.venv/bin/python -m proc2d run examples/deck_oxidation_implant_anneal.yaml --out outputs/run_p2`
- 결과: `Done. Grid=(201, 401), exports=12`
- 생성물: `material.vtk`, `tox_vs_x.csv`, `tox_vs_x.png` 포함

3) Arrhenius schedule deck

- 명령: `.venv/bin/python -m proc2d run examples/deck_schedule_anneal.yaml --out outputs/run_sched`
- 결과: `Done. Grid=(201, 401), exports=2`

---

## 7) 주의사항 / 리스크

1. 작업트리가 clean 상태가 아님

- `scripts/setup_venv.sh` 변경 존재
- `The`, `This` 빈 파일이 untracked로 존재

2. 출력 디렉터리 재사용 시 이전 결과 파일이 남을 수 있음

- 같은 `--out` 경로를 재사용하면 이전 산출물이 섞여 보일 수 있으므로 필요 시 수동 정리 권장

3. GUI 실행 시 포트 충돌 가능

- `8502` 사용 중이면 `--server.port`를 다른 값으로 지정 필요

---

## 8) 권장 다음 액션

P0:

1. 불필요 파일 정리: `The`, `This`
2. `scripts/setup_venv.sh` 변경 내용 검토 후 커밋 여부 결정
3. 결과 디렉터리 운영 원칙 통일(실행 전 비우기 또는 실행별 별도 outdir)

P1:

1. 보고서와 함께 릴리즈 노트/CHANGELOG 동기화
2. CI에서 예제 deck smoke test 추가 여부 검토

---

## 9) 체크리스트

- [x] Core 파이프라인 동작 확인
- [x] P2(oxidation/Arrhenius/tox/material) 동작 확인
- [x] Analyze/History/VTK 출력 확인
- [x] GUI 기능 존재 확인(README 기준)
- [x] 테스트 통과(12 passed)
- [x] 예제 deck 3종 실행 확인
- [ ] 최종 커밋/푸시 및 원격 CI 확인
