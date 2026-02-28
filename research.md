# proc2d 모듈 심층 리서치 보고서

작성일: 2026-02-25  
대상: `proc2d` 패키지 및 연관 예제/테스트/운영 파일

## 1) 조사 범위와 방법

- 조사 범위
  - 코어 소스: `proc2d/*.py`
  - 테스트: `tests/*.py`
  - 예제 deck: `examples/*.yaml`
  - 실행/운영 보조: `README.md`, `pyproject.toml`, `scripts/*.sh`, `.github/workflows/ci.yml`
- 방법
  - 정적 코드 리딩(함수 단위 동작, 입출력 형태, 경계조건 처리, 예외 처리, 단위 일관성) 중심
  - deck 파이프라인과 GUI 파이프라인을 각각 추적
  - 테스트 코드로 의도된 물리/수치 검증 포인트 역추적
- 실행 검증 시도 결과
  - 현재 환경에서 `numpy`, `pytest` 미설치로 실행형 검증은 실패
  - 실패 로그 요지
    - `python3 -m proc2d run ...` -> `ModuleNotFoundError: No module named 'numpy'`
    - `python3 -m pytest -q` -> `No module named pytest`
  - 따라서 본 보고서는 **실행 결과가 아닌 코드 기반 심층 분석**이다.

---

## 2) 프로젝트 한 줄 요약

`proc2d`는 반도체 공정 2D 단면에서 `mask -> oxidation(optional) -> implant -> anneal -> analyze -> export` 순서를 deck(YAML)로 기술해 시뮬레이션하는 도구다. 핵심 PDE는 2D 확산식이며, 암시적(Backward Euler) 시간 적분과 희소 선형계 해법으로 안정적으로 계산한다.

---

## 3) 패키지 구조와 역할 분해

- `proc2d/units.py`
  - um<->cm 변환, 양수/비음수 검증 유틸
- `proc2d/grid.py`
  - 정규 구조 격자(`Grid2D`) 정의, 좌표/간격/인덱스 유틸
- `proc2d/mask.py`
  - 1D 개구(opening) 마스크 생성 + lateral Gaussian smoothing
- `proc2d/oxidation.py`
  - Deal-Grove 기반 산화막 두께 갱신, 표면 이동, 재료맵 생성
- `proc2d/implant.py`
  - separable 2D Gaussian 주입 모델(산화막 두께 반영)
- `proc2d/diffusion.py`
  - 확산 연산자 조립, top mixed BC(Neumann/Robin/Dirichlet), 암시적 anneal, history
- `proc2d/metrics.py`
  - 질량/피크/접합깊이/수평폭/iso area/sheet dose 지표
- `proc2d/io.py`
  - npy/csv/png/vtk 및 metrics/history/tox 결과 저장
- `proc2d/deck.py`
  - YAML deck 파싱/검증/실행 오케스트레이션(실질적 엔진)
- `proc2d/cli.py`, `proc2d/__main__.py`
  - CLI 엔트리 (`python -m proc2d run ...`)
- `proc2d/gui_app.py`, `proc2d/gui.py`, `proc2d/gui_script.py`
  - Streamlit GUI 실행과 인터랙티브 시뮬레이션

---

## 4) 데이터 모델/단위 체계

### 4.1 좌표/배열 규약

- 좌표
  - `x`: lateral
  - `y`: depth (`y=0`이 top surface)
- 필드 shape
  - 농도장 `C` shape는 항상 `(Ny, Nx)`
- flatten 규약
  - row-major, `flat_index(j,i)=j*Nx+i`

### 4.2 단위

- 입력 길이: um
- 내부 확산 계산 길이: cm (`um_to_cm` 사용)
- 농도: `cm^-3`
- 2D 단면 적분량(암묵적 z 두께 1 cm): `cm^-1`
  - 예: `mass = sum(C) * dx_cm * dy_cm`

### 4.3 상태 구조 (`SimulationState`)

- 핵심 필드
  - `grid`, `C`
  - `mask_eff` (Nx, [0,1])
  - `tox_um` (Nx)
  - `materials` ((Ny,Nx), `0=Si`, `1=SiO2`)
  - `metrics`, `history`, `exports`
- 산화 관련 상태는 `_ensure_state_oxide_fields`로 강제 일관성 유지

---

## 5) 물리/수치 모델 상세

## 5.1 Mask 모델 (`mask.py`)

- `build_mask_1d(x_um, openings_um)`
  - opening 구간 `[x0,x1]`에 대해 binary mask 생성
  - 도메인 밖 구간은 자동 clip
  - 구간 경계 포함(`x>=left & x<=right`)
- `smooth_mask_1d(mask, sigma_lat_um, dx_um)`
  - `sigma_px = sigma_lat_um/dx_um`로 변환 후 `gaussian_filter1d(mode="nearest")`
  - 결과는 `[0,1]` clip
- 설계 포인트
  - mask를 완전 binary로만 쓰지 않고 BC 혼합비로도 사용 가능

## 5.2 Implant 모델 (`implant.py`)

- 깊이 profile
  - `g(y)=dose/(sqrt(2*pi)*dRp) * exp(-0.5*((y-Rp)/dRp)^2)`
- 2D separable 확장
  - `dC(y,x)=g(y_eff)*mask_eff(x)`
  - `y_eff = y_um - tox_um[x]` (산화막 두께만큼 peak가 아래로 이동)
  - `y_eff<0`는 0으로 절단(산화막 위쪽 주입 제거)
- 검증 포인트
  - `tests/test_oxidation_p2.py`에서 산화막이 두꺼운 칼럼의 peak가 정확히 아래로 이동하는지 확인

## 5.3 Oxidation 모델 (`oxidation.py`)

- Deal-Grove 업데이트
  - `tau=(tox_old^2 + A*tox_old)/B`
  - `tox_new=(-A + sqrt(A^2 + 4*B*(time+tau)))/2`
  - `tox_new >= tox_old` 강제
  - `B=0`이면 성장 없음
- 적용 영역
  - `apply_on`: `all | open | blocked`
  - open/blocked 판정은 `mask_eff>0.5` 기준(binary-like)
- 표면 이동
  - `delta_tox = tox_new - tox_old`
  - `delta_out = delta_tox * (1 - 1/gamma)`
  - 각 x 칼럼에서 `C_new(y)=C_old(y-delta_out[x])` 선형보간
- 재료맵
  - `materials = (y < tox(x))` -> SiO2(1), else Si(0)
- 도펀트 처리
  - `consume_dopants=True`이면 산화막 영역 농도 0

## 5.4 Diffusion/Anneal 모델 (`diffusion.py`)

- PDE
  - `∂C/∂t = ∇·(D∇C)`
- 공간 이산화
  - regular grid finite-volume 스타일
  - face D는 harmonic mean 사용
- 경계조건
  - 좌/우/하단: Neumann(0 flux)
  - 상단(top): open/blocked 혼합
    - 혼합 가중치 `open_frac = clip(mask_eff,0,1)`
    - open type
      - `neumann`: 항 없음
      - `robin`: `-D dC/dn = h(C-Ceq)`
      - `dirichlet`: 데모성 fallback 구현
    - blocked는 현재 `neumann`만 허용
- 연산자 형태
  - `dC/dt = A*C + b`
  - 시간적분: Backward Euler
  - `(I-dt*A)C_{n+1} = C_n + dt*b`
- 구현 세부
  - 반복 step에서는 LU factorization(`splu`) 재사용으로 성능 확보
  - 잔여 시간(`rem`)은 마지막에 별도 선형해결

## 5.5 Anneal History

- `anneal_implicit_with_history`에서 선택적 기록
  - 기본 필드: `time_s, mass, peak_cm3, flux_out, residual`
  - `residual = (M_now - M_prev)/dt + flux_out` (Robin에서만 의미)
  - 초기 레코드 `t=0` 포함, residual은 `NaN`
  - 기록 간격 `record_every_s`는 독립 설정 가능

---

## 6) Deck 엔진 동작(핵심)

## 6.1 로딩/검증

- `load_deck`
  - 파일 존재/파싱 실패/빈 deck 처리
  - 최상위 mapping 강제
- 공통 유틸
  - `_required`, `_to_float`, `_to_int`, `_opt_mapping`
  - 예외는 `DeckError`로 표준화

## 6.2 출력 경로 결정 규칙

- `out_override`가 있으면 항상 우선
- 없으면
  - 해당 step의 `outdir` 우선
  - step `outdir` 없으면 deck 내 첫 `export` step의 `outdir`
  - 그것도 없으면 `outputs/run`
- 상대경로 처리
  - `out_override` 상대경로는 현재 작업 디렉터리 기준 `resolve()`
  - deck 내부 outdir 상대경로는 **deck 파일 위치 기준**

## 6.3 step별 실행 규약

### mask step

- 필수: `openings_um`
- 옵션: `sigma_lat_um`(default 0)
- 결과: `state.mask_eff`

### oxidation step

- 필수: `time_s`, `A_um`, `B_um2_s`
- 옵션: `model=deal_grove`, `gamma=2.27`, `apply_on=all`, `consume_dopants=True`, `update_materials=True`
- 특수: `tox_init_um` 제공 시 초기 tox(0일 때만) 일괄 설정
- 안전장치: `max(tox) > Ly_um`이면 즉시 실패

### implant step

- 필수: `dose_cm2`, `Rp_um`, `dRp_um`
- mask 없으면 full-open 자동
- tox 반영 주입 후 `state.C += dC`

### anneal step

- 확산계수 지정 두 경로
  - `D_cm2_s` + `total_t_s`
  - `diffusivity.model=arrhenius`
    - 단일 `T_C` 또는 `schedule[{t_s,T_C},...]`
    - schedule 사용 시 `total_t_s`가 있으면 합계와 일치 강제
- 공통 필수: `dt_s`
- 산화막 장벽
  - `oxide.D_scale`로 SiO2 영역 D 축소
  - `cap_eps_um`로 top BC gate (`tox<=cap_eps`인 칼럼만 open)
- history
  - `record.enable`, `record.every_s`, `save_csv`, `save_png`
  - 다중 segment schedule에서는 segment 간 중복 t=0 레코드 제거 후 시간 오프셋 병합

### analyze step

- 기본 보고 항목
  - `total_mass_cm1`, `peak`
- 옵션
  - `silicon_only`: oxide 영역 0 처리 후 평가
  - `junctions[]`, `laterals[]`, `iso_area_threshold_cm3`
  - `sheet_dose.save_csv`
  - `save.json/csv` (기본 true)

### export step

- 형식: `npy/csv/png/vtk`
- `csv` linecut 미지정 시 중앙 vertical 기본 1개 자동 생성
- `vtk` 선택 시
  - `C.vtk`
  - material map이 있으면 `material.vtk`
  - `plot.log10=true`면 `C_log10.vtk`
- `extra.tox_csv/tox_png`로 tox profile 추가 출력

---

## 7) 지표(metrics) 계산 로직 상세

- `total_mass`: 전영역 적분(`cm^-1`)
- `peak_info`: 최대값과 위치 `(i,j,x_um,y_um)`
- `sheet_dose_vs_x`: `sum_y C * dy_cm` (`cm^-2`)
- `junction_depth_1d`
  - threshold crossing을 선형보간
  - `mode=first|last` 지원
  - exact threshold 점에서 중복 crossing 제거 로직 포함
- `junction_depth`
  - 요청 x를 nearest grid로 snap 후 1D depth 추정
- `lateral_extents_at_y`
  - 요청 y를 nearest grid로 snap
  - `C>=threshold` 구간을 x 세그먼트로 계산
  - 폭 계산 시 half-cell 경계 보정
- `iso_contour_area`
  - 엄밀 contour 적분이 아니라 cell-count 근사

---

## 8) I/O 및 파일 포맷 세부

## 8.1 일반 출력

- `C.npy`
  - float 배열 원본 저장
- linecut csv
  - vertical: `y_um,C_cm3`
  - horizontal: `x_um,C_cm3`
  - 요청값/실사용 snap값 메타 포함
- `C.png`
  - `plot.log10` 모드 시 `log10(clip(C,floor,None))`
  - cmap: `inferno`

## 8.2 metrics/history 출력

- `metrics.json`
  - 계층 구조 유지
- `metrics.csv`
  - 중첩 dict를 dot-path key로 flatten
  - list 값은 JSON 문자열로 직렬화
- `history.csv`, `history.png`
  - mass/flux/residual 시계열

## 8.3 VTK 출력

- legacy VTK ASCII, `STRUCTURED_POINTS`
- `DIMENSIONS Nx Ny 1`, `SPACING dx_um dy_um 1`
- scalar
  - doping: `C.vtk`
  - log10 doping: `C_log10.vtk`(옵션)
  - material map: `material.vtk`(옵션)

---

## 9) GUI 내부 동작 분석

## 9.1 기본 철학

- GUI는 deck 엔진을 직접 호출하지 않고, 유사 파이프라인을 `run_simulation`에서 별도로 수행
- 사용자가 즉시 결과 확인/비교/다운로드 가능하도록 결과 객체를 세션에 저장

## 9.2 GUI 파이프라인

- 파라미터 수집 -> `Grid2D` 생성 -> 초기 `C/tox/materials`
- mask 생성/스무딩
- optional oxidation 1회
- implant
- anneal
  - 상단 BC 타입 선택(robin/neumann/dirichlet)
  - arrhenius는 단일 온도 모델만 지원(스케줄 없음)
- export 결과 저장
- optional metrics/history/vtk/zip

## 9.3 탭별 기능

- Map
  - 농도 heatmap + mask open 표시 + tox 곡선
- Linecuts
  - vertical/horizontal 선택, linear/log10 전환, CSV 다운로드
- Metrics
  - 요약 metric + JSON 표시 + 파일 다운로드
- History
  - 3패널 그래프 + CSV/PNG 다운로드
- Compare
  - 최근 2회 실행 A/B 맵 및 linecut overlay
  - `store_full_c` 켜면 동일 color scale 고정 비교 가능
- Artifacts
  - 생성 파일 목록 및 개별 다운로드(vtk/tox/png/zip)

## 9.4 GUI와 deck 엔진의 차이점(중요)

- GUI
  - anneal arrhenius schedule 미지원(단일 T)
  - analyze는 고정 템플릿 metric 계산(임계값 1e17/1e18 등)
- deck 엔진
  - schedule 지원
  - analyze 항목을 deck에서 자유 구성

---

## 10) 테스트 체계로 본 신뢰성 분석

- `test_mass_conservation.py`
  - all-Neumann에서 질량 보존(상대오차 매우 엄격)
- `test_symmetry.py`
  - 중심 opening에서 x-대칭 보존 검증
- `test_1d_limit.py`
  - x-균일 조건에서 x-균일성 유지(1D limit)
- `test_metrics.py`
  - mass/peak/sheet dose/junction/lateral 유틸 기능 정확성
- `test_history.py`
  - history.csv 생성 및 시간 단조 증가
- `test_vtk_writer.py`
  - VTK 헤더/토큰/데이터 포인트 수 확인
- `test_oxidation_p2.py`
  - Deal-Grove 단조성
  - 표면 이동 보간 동작
  - 산화막 두께에 따른 implant peak 이동
  - Arrhenius 온도 증가 -> D 증가
  - deck 기반 tox/material export 존재 확인

판단: 단순 smoke 수준을 넘어, 핵심 물리 불변량(질량/대칭/한계거동)과 신규 기능(history/vtk/oxidation)의 회귀 포인트가 명확히 잡혀 있다.

---

## 11) 설계 강점과 리스크

## 11.1 강점

- 모듈 경계가 명확하다(physics / deck / io / gui)
- 입력 검증이 촘촘하며 오류 메시지가 문맥 친화적
- 확산 solver에서 LU 재사용으로 반복 step 비용 절감
- 산화막-재료맵-확산장벽-BC gate가 일관된 P2 확장 경로를 형성
- 출력 형식이 실사용 친화적(npy/csv/png/vtk + tox/history/metrics)

## 11.2 리스크/주의점

- GUI와 deck 엔진에 로직 중복이 크다
  - 기능 추가 시 동기화 누락 위험
- `iso_contour_area`는 contour 기반이 아니라 cell-count 근사
- oxidation의 open/blocked 판정은 `mask>0.5` 이진화 기준
  - mask edge fractional 효과를 산화에는 반영하지 않음
- material map 기준이 `y < tox`라 경계 `y==tox`는 Si로 처리
- 실행 환경 의존성이 강함(현재 환경처럼 `numpy/pytest` 부재 시 즉시 실행 불가)

---

## 12) 실무 관점 개선 제안

1. GUI 계산 경로를 deck 엔진 호출 방식으로 통합
   - 중복 제거, 기능 편차(예: schedule) 해소
2. analyze/metrics 확장
   - contour 기반 면적 추정(보간) 옵션 추가
3. BC/산화 coupling 고도화
   - oxidation에도 fractional mask 가중 적용 옵션 검토
4. 환경 재현성 강화
   - 최소 실행용 `requirements` 가이드와 startup self-check 명령 제공

---

## 13) 결론

이 모듈은 MVP를 넘어, 산화(Deal-Grove), 산화막 기반 주입 깊이 이동, 산화막 장벽 확산, mixed top BC, history/metrics/vtk까지 포함하는 **완성도 높은 2D 공정 단면 시뮬레이터**로 구성되어 있다. 아키텍처상 핵심 엔진(`deck.py`)의 설계와 수치해석 구현(`diffusion.py`)은 일관성이 좋고 검증 포인트도 적절하다. 현재 가장 큰 구조적 과제는 GUI-엔진 중복이며, 이를 통합하면 유지보수성과 기능 일관성이 크게 개선될 것으로 판단된다.
