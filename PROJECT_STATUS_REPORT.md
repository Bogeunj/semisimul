# proc2d 프로젝트 현황 보고서

작성일: 2026-02-20  
대상 경로: `/mnt/c/Users/user/semisimul`

## 1) 문서 목적

이 문서는 `proc2d` 코드베이스의 **현재 구현 상태(기능/검증/운영 기반)**를 한 번에 파악하기 위한 최신 리포트다.  
특히 이번 스프린트 목표였던 Core(Analyze/History/VTK), GUI 통합, Ops 정리 반영 여부를 중심으로 정리한다.

---

## 2) 요약 (Executive Summary)

이번 업데이트 기준 결론:

- 기존 MVP(`mask -> implant -> anneal -> export`)는 하위호환 유지
- Core 확장 3종 반영 완료
  - Analyze/Metrics step
  - Anneal History 기록
  - VTK export
- GUI 확장 반영 완료
  - Metrics/History/VTK/ZIP 다운로드
  - 최근 2회 Before/After Compare 탭
- Ops 파일 반영 완료
  - `.gitignore` 보강
  - `scripts/setup_user_install.sh`, `scripts/setup_venv.sh`
  - `.github/workflows/ci.yml`
- 검증 통과
  - `pytest`: **7 passed**
  - `deck_basic` 실행 성공
  - `deck_analysis_history_vtk` 실행 성공(11개 산출물)
  - GUI Streamlit 스모크 기동 성공

---

## 3) 프로젝트 개요

- 프로젝트 성격: 반도체 공정 2D 단면(Process 2D cross-section) 시뮬레이터
- 패키지명/버전: `proc2d / 0.1.0`
- 언어: Python 3.10+
- 빌드: `pyproject.toml` + setuptools editable install
- 실행 인터페이스:
  - CLI: `python3 -m proc2d run ...`
  - GUI: `python3 -m streamlit run proc2d/gui_script.py`

---

## 4) 저장소/형상관리 상태

### 4.1 Git 상태

- 현재 경로는 Git 저장소임
- 브랜치: `main` (원격 `origin/main` 트래킹)
- 원격: `https://github.com/Bogeunj/semisimul.git`
- 현재 작업트리 상태: **변경/신규 파일 존재(아직 커밋 전)**

현재 변경 파일(요약):

- 수정: `.gitignore`, `README.md`, `proc2d/deck.py`, `proc2d/diffusion.py`, `proc2d/gui_app.py`, `proc2d/io.py`
- 신규: `.github/workflows/ci.yml`, `proc2d/metrics.py`, `examples/deck_analysis_history_vtk.yaml`, `scripts/*.sh`, `tests/test_metrics.py`, `tests/test_vtk_writer.py`, `tests/test_history.py`

### 4.2 구조 요약

- 소스: `proc2d/`
- 테스트: `tests/`
- 예제 deck: `examples/deck_basic.yaml`, `examples/deck_analysis_history_vtk.yaml`
- 실행 출력 예시: `outputs/run1/`, `outputs/run2/`

---

## 5) 기능 구현 현황 (요구사항 매핑)

## 5.1 Core-1: Analyze/Metrics step

상태: **완료**

구현 내용:

- 신규 모듈 `proc2d/metrics.py`
  - `total_mass(C, grid)`
  - `peak_info(C, grid)`
  - `sheet_dose_vs_x(C, grid)`
  - `junction_depth_1d(profile, y, threshold, mode)`
  - `junction_depth(C, grid, x_um, threshold_cm3)`
  - `lateral_extents_at_y(C, grid, y_um, threshold_cm3)`
  - `iso_contour_area(C, grid, threshold_cm3)`

- `deck.py`에 `type: analyze` step 추가
  - 하위호환 유지: analyze step이 없으면 기존 동작 동일

- 출력 파일
  - `metrics.json`
  - `metrics.csv`
  - `sheet_dose_vs_x.csv` (옵션)

- 내부 상태 저장
  - `SimulationState.metrics`에 결과 저장

## 5.2 Core-2: Anneal History 기록

상태: **완료**

구현 내용:

- `diffusion.py`
  - `top_flux_out(...)` 추가
  - `anneal_implicit_with_history(...)` 추가

- 기록 필드(최소 요구 충족)
  - `time_s`
  - `mass`
  - `peak_cm3`
  - `flux_out`
  - `residual`

- `deck.py` anneal step 확장
  - `record.enable`, `record.every_s`, `save_csv`, `save_png` 지원
  - 출력: `history.csv`, `history.png`

- 하위호환
  - 기존 `anneal_implicit(...)` 인터페이스 유지

## 5.3 Core-3: VTK export

상태: **완료**

구현 내용:

- `io.py`
  - `save_vtk_structured_points(...)` 추가
  - `export_results(..., formats=...)`에 `vtk` 지원 추가

- 출력
  - `C.vtk`
  - `C_log10.vtk` (`plot.log10=true`일 때)

- 포맷
  - legacy VTK ASCII, `STRUCTURED_POINTS`
  - `DIMENSIONS Nx Ny 1`
  - `SPACING dx_um dy_um 1`

## 5.4 GUI-1: Core 기능 통합

상태: **완료**

반영 사항:

- Run 옵션 토글
  - Compute metrics
  - Record anneal history
  - Export VTK
  - Download all outputs as ZIP

- 결과 탭
  - `Map` (mask overlay 포함)
  - `Linecuts` (linear/log10)
  - `Metrics` (json 표시 + 다운로드)
  - `History` (그래프 + CSV 다운로드)
  - `Artifacts` (VTK/ZIP/C.png 포함)

- 실행시간 표시
  - `time.perf_counter()` 기반

## 5.5 GUI-2: 최근 2회 Before/After 비교

상태: **완료**

반영 사항:

- 세션에 최근 2회 run 기록 유지
- Compare 탭에서 A/B 맵, linecut overlay, metrics diff 표 제공
- 메모리 고려 옵션
  - 기본: full C 미저장
  - 옵션: `Store full C in session` 켜면 full C 저장

## 5.6 Ops-1: Git/CI 기반 파일

상태: **완료(파일 반영)**

- `.gitignore` 보강
  - `__pycache__`, `.pytest_cache`, `*.egg-info`, `.venv`, `outputs`, `*.vtk`, `*.npy`, `.streamlit` 등

- CI 파일 추가
  - `.github/workflows/ci.yml`
  - Python 3.10/3.11 매트릭스, `pip install -e ".[dev]"`, `pytest -q`

## 5.7 Ops-2: 환경 재현성 스크립트

상태: **완료**

- `scripts/setup_user_install.sh`
  - `--user --break-system-packages` 루트 자동화
  - `~/.local/bin` PATH 안내

- `scripts/setup_venv.sh`
  - venv 생성 + `-e ".[dev,gui]"` 설치
  - venv 미지원 환경 힌트 제공

---

## 6) 테스트/검증 현황

### 6.1 테스트 구성

기존 + 신규 포함 총 7개 통과:

- 기존:
  - `tests/test_mass_conservation.py`
  - `tests/test_symmetry.py`
  - `tests/test_1d_limit.py`
- 신규:
  - `tests/test_metrics.py`
  - `tests/test_vtk_writer.py`
  - `tests/test_history.py`

### 6.2 최신 실행 결과

1) 단위 테스트

- 명령: `python3 -m pytest -q`
- 결과: `....... [100%]` (**7 passed**)

2) 기본 deck 실행

- 명령: `python3 -m proc2d run examples/deck_basic.yaml --out outputs/run1`
- 결과: 정상 완료, exports=4

3) 확장 deck 실행(analyze/history/vtk)

- 명령: `python3 -m proc2d run examples/deck_analysis_history_vtk.yaml --out outputs/run2`
- 결과: 정상 완료, exports=11
- 생성 확인:
  - `C.npy`, `C.png`, `C.vtk`, `C_log10.vtk`
  - `metrics.json`, `metrics.csv`, `sheet_dose_vs_x.csv`
  - `history.csv`, `history.png`
  - linecut CSV 2개

4) GUI 스모크

- 명령: `python3 -m streamlit run proc2d/gui_script.py --server.headless true --server.port 8503`
- 결과: 로컬 URL 출력 확인(기동 성공)

---

## 7) 문서화 상태

상태: **업데이트 완료**

- `README.md`에 반영:
  - analyze step / history 옵션 / vtk 설명
  - GUI 신규 탭/다운로드 기능
  - venv 가능/불가 설치 루트
  - 예제 deck 2종 실행 가이드

---

## 8) 남은 리스크/주의사항

1. 현재 변경사항은 아직 미커밋 상태
   - 커밋/푸시 전 누락 파일 점검 필요

2. GUI 비교 기능은 옵션 기반 full C 저장 시 메모리 사용량 증가 가능

3. CI 파일은 추가되었지만 실제 GitHub Actions 동작은 push 후 확인 필요

4. Streamlit 포트 충돌 가능
   - 예: 8502 사용중이면 다른 포트(8503 등) 사용 필요

---

## 9) 권장 다음 액션

P0:

1. 변경사항 커밋
2. 원격 push 후 CI 동작 확인
3. GUI 수동 점검
   - Metrics/History/Compare 탭
   - ZIP/VTK 다운로드

P1:

1. Compare 탭의 large-grid 성능 최적화
2. Analyze 지표 확장(요구 시 profile overlay 등)

---

## 10) 체크리스트 (현재 시점)

- [x] Core-1 Analyze/Metrics
- [x] Core-2 Anneal History
- [x] Core-3 VTK export
- [x] GUI-1 Metrics/History/VTK/ZIP 통합
- [x] GUI-2 최근 2회 Compare
- [x] Ops-1 `.gitignore` + CI 파일
- [x] Ops-2 setup 스크립트 2종
- [x] 테스트 통과 (7 passed)
- [x] `deck_basic` 실행 검증
- [x] `deck_analysis_history_vtk` 실행 검증
- [ ] 최종 커밋/푸시 및 CI 원격 확인
