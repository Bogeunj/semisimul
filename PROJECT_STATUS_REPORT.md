# proc2d 프로젝트 현황 보고서

작성일: 2026-03-01  
대상 경로: `/home/bogeun/projects/semisimul`  
기준 커밋: `e039908` (2026-02-28 21:47:30 +0900)

## 1) 문서 목적

이 문서는 `proc2d` 저장소의 현재 구현 범위, 검증 결과, 작업트리 상태를 한 번에 확인하기 위한 최신 리포트다.

---

## 2) 요약 (Executive Summary)

현재 기준 결론:

- Core 파이프라인(`mask -> oxidation(optional) -> implant -> anneal -> analyze -> export`)은 예제 deck 3종에서 정상 실행됨
- CLI는 현재 `run` 서브커맨드만 제공함 (`python3 -m proc2d --help` 기준)
- GUI는 코드 기준으로 탭/다운로드/최근 2회 비교 기능을 유지함
- Git 작업트리는 clean 상태임 (`## main...origin/main`)
- 예제 deck 실행 확인:
  - `deck_analysis_history_vtk`: `Done. Grid=(201, 401), exports=12`
  - `deck_oxidation_implant_anneal`: `Done. Grid=(201, 401), exports=12`
  - `deck_schedule_anneal`: `Done. Grid=(201, 401), exports=2`
- 테스트는 통과 상태가 아님
  - `.venv/bin/python -m pytest -q` 실행 시 collection 단계에서 ImportError 2건 발생
  - 원인: `proc2d.deck`에서 `run_deck_data`를 import할 수 없음
- `selfcheck`도 현재 동작 불가 상태
  - `python3 -m proc2d selfcheck`는 CLI 미지원(`run`만 지원)
  - `proc2d.selfcheck` 모듈 직접 import도 동일한 `run_deck_data` ImportError로 실패

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
- 현재 작업트리: clean
  - 상태 출력: `## main...origin/main`
- 참고: linked worktree 브랜치 `a`, `b` 존재 (`semisimul_a`, `semisimul_b`)

### 4.2 디렉터리 구성 요약

- 소스: `proc2d/`
- 테스트: `tests/` (현재 `tests/**/*.py` 20개)
- 예제 deck:
  - `examples/deck_basic.yaml`
  - `examples/deck_analysis_history_vtk.yaml`
  - `examples/deck_oxidation_implant_anneal.yaml`
  - `examples/deck_schedule_anneal.yaml`

---

## 5) 기능 구현 현황

### 5.1 실행 아키텍처

- deck 실행 경로는 `proc2d/deck.py`의 `run_deck`/`run_simulation_payload`를 통해 공통 서비스(`SimulationService`)를 사용
- 기본 step handler 등록:
  - `mask`, `oxidation`, `implant`, `anneal`, `analyze`, `export`
- GUI는 deck payload 생성 후 공용 실행 경로(`run_simulation_payload`)를 호출
- 주의: `proc2d/engine.py`는 `run_deck_data`를 import하도록 남아 있어 현재 ImportError를 유발함

### 5.2 시뮬레이션 코어

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

### 5.3 분석/출력

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

### 5.4 GUI

- 실행/파라미터 조정 + 결과 탭 제공
  - `Map`, `Linecuts`, `Metrics`, `History`, `Compare`, `Artifacts`
- 다운로드/보조 기능
  - VTK, ZIP 다운로드
  - 최근 2회 before/after 비교
  - full C 세션 저장(옵션)

### 5.5 옵션 구현/연결 상태 메모

- 코드 레벨 옵션 구현 확인:
  - `metrics.iso_contour_area(..., method="tri_linear")`
  - oxidation `mask_weighting` (`binary`/`time_scale`)
  - diffusion `cap_model` (`hard`/`exp`)
- 단, pipeline step wiring은 일부 구간에서 보강 필요 가능성 있음
  - 현재 장애(`run_deck_data` ImportError) 해결 후 전체 회귀 검증 필요

---

## 6) 테스트/검증 현황 (2026-03-01 기준)

### 6.1 CLI 인터페이스 확인

- 명령: `python3 -m proc2d --help`
- 결과: `usage: proc2d [-h] {run} ...`
- 현재 지원 서브커맨드: `run`만 노출

### 6.2 테스트 실행

- 명령: `.venv/bin/python -m pytest -q`
- 결과: **실패 (collection 단계 ImportError 2건)**

핵심 에러:

1) `tests/test_engine_and_gui_bridge.py`

- `from proc2d.deck import load_deck, run_deck_data`
- `ImportError: cannot import name 'run_deck_data' from 'proc2d.deck'`

2) `tests/test_selfcheck.py`

- `proc2d.selfcheck -> proc2d.engine -> from .deck import ... run_deck_data`
- 동일 ImportError 발생

보충:

- marker 필터를 적용한 실행(`-m "not integration and not adapter and not slow"`)도 동일 이유로 collection 단계에서 중단됨

### 6.3 selfcheck 상태

1) CLI 경로:

- 명령: `python3 -m proc2d selfcheck`
- 결과: `invalid choice: 'selfcheck' (choose from 'run')`

2) 모듈 직접 경로:

- 명령: `.venv/bin/python -c "from proc2d.selfcheck import run_selfcheck; print(run_selfcheck())"`
- 결과: `run_deck_data` ImportError로 실패

### 6.4 예제 deck 실행

검증 명령:

1) `.venv/bin/python -m proc2d run examples/deck_analysis_history_vtk.yaml --out outputs/report_check_run2`

- 결과: `Done. Grid=(201, 401), exports=12`
- 생성물(12): `C*.vtk`, `history.*`, `metrics.*`, linecut CSV, `sheet_dose_vs_x.csv`

2) `.venv/bin/python -m proc2d run examples/deck_oxidation_implant_anneal.yaml --out outputs/report_check_run_p2`

- 결과: `Done. Grid=(201, 401), exports=12`
- 생성물(12): `C*.vtk`, `material.vtk`, `metrics.*`, `tox_vs_x.csv`, `tox_vs_x.png` 등

3) `.venv/bin/python -m proc2d run examples/deck_schedule_anneal.yaml --out outputs/report_check_run_sched`

- 결과: `Done. Grid=(201, 401), exports=2`
- 생성물(2): `C.npy`, `C.png`

---

## 7) 주의사항 / 리스크

1. `run_deck_data` API 불일치로 테스트/자체점검 경로가 깨져 있음

- `proc2d.engine` 및 일부 테스트가 구 심볼(`run_deck_data`)을 참조
- 현재 `proc2d.deck` 공개 API는 `run_simulation_payload` 중심

2. 문서-실행 경로 불일치 (`selfcheck`)

- README에는 `python -m proc2d selfcheck`가 안내되어 있으나
- 현재 CLI 구현은 `run`만 지원

3. 옵션 기능의 end-to-end 검증 공백

- `tri_linear`, `mask_weighting`, `cap_model` 등은 코드 조각 단위 구현이 보임
- 그러나 전체 회귀 테스트가 ImportError에서 중단되어 현재 세션에서 완전 검증이 불가함

4. 출력 디렉터리 재사용 시 이전 결과 파일 혼합 가능성

- 같은 `--out` 경로 재사용 시 산출물 관리 원칙 필요

---

## 8) 권장 다음 액션

P0:

1. `run_deck_data` 호환성 복구 또는 호출부 전환
   - 선택 A: `proc2d.deck`에 호환 alias 추가
   - 선택 B: `proc2d.engine`/테스트에서 `run_simulation_payload` 사용으로 일괄 전환
2. CLI에 `selfcheck` 서브커맨드 연결 또는 README selfcheck 문구 정정
3. 위 1~2 반영 후 `pytest -q` 재실행, 실패/통과 수치 갱신
4. selfcheck + 예제 deck smoke를 CI/로컬 표준 검증 시퀀스로 고정

P1:

1. `iso_area.method`, `oxidation.mask_weighting`, `anneal cap_model`의 deck->step->core wiring 점검
2. `PROJECT_STATUS_REPORT.md`, `README.md`, `plan.md` 상태 설명 동기화

---

## 9) 체크리스트

- [x] Core 파이프라인 예제 deck 3종 실행 확인
- [x] Analyze/History/VTK/tox 출력 산출물 확인
- [x] GUI 탭/다운로드/비교 기능 존재 확인(코드 기준)
- [x] Git 작업트리 clean 상태 확인
- [ ] `pytest` 전체 통과
- [ ] `proc2d selfcheck` CLI 경로 정상화
- [ ] `proc2d.selfcheck` 모듈 경로 정상화
- [ ] 문서(README)와 실제 실행 경로 일치화
- [ ] 원격 CI까지 포함한 최종 검증
