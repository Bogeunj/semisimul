# proc2d

반도체 공정 2D 단면(Process 2D cross-section) 시뮬레이터입니다.

현재 구현 범위(P2 포함):

- 1D mask(openings) + lateral Gaussian smoothing
- 2D separable implant (oxide 두께 `tox(x)` 반영)
- oxidation step (Deal-Grove, 표면 이동, Si/SiO2 material map)
- 2D implicit diffusion anneal
- anneal diffusivity: 상수 `D_cm2_s` 또는 Arrhenius(+schedule)
- mixed top BC (open: Robin/Neumann/Dirichlet, blocked: Neumann)
- oxide barrier: `oxide.D_scale`, `cap_eps_um`, `cap_model`로 top open gate 제어
- deck step 파이프라인(권장 순서): `mask -> oxidation(optional) -> implant -> anneal -> analyze -> export`
- 출력: `npy`, `csv`, `png`, `vtk(ParaView)`, `tox_vs_x.(csv/png)`, `material.vtk`
- GUI(Streamlit): 파라미터 조정/실행/맵/라인컷/metrics/history/compare/다운로드

---

## 설치

### A) venv 가능 환경 (권장)

```bash
./scripts/setup_venv.sh
```

또는 수동:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,gui]"
python -m proc2d selfcheck
```

### B) venv 어려운 환경 (WSL/시스템 Python)

```bash
./scripts/setup_user_install.sh
```

또는 수동:

```bash
python3 -m pip install --user --break-system-packages -e ".[dev,gui]"
python3 -m proc2d selfcheck
```

`~/.local/bin`이 PATH에 없으면 아래를 쉘 설정에 추가하세요.

```bash
export PATH="$HOME/.local/bin:$PATH"
```

---

## 빠른 실행

CLI:

```bash
python3 -m proc2d run examples/deck_basic.yaml --out outputs/run1
python3 -m proc2d selfcheck
```

GUI:

```bash
proc2d-gui --server.port 8502
```

또는:

```bash
python3 -m streamlit run proc2d/gui_script.py --server.port 8502
```

---

## 모듈 구조 (리팩터링 반영)

- 공통 실행 엔진: `proc2d/deck.py`의 step 핸들러를 `proc2d/services/simulation_service.py` + `proc2d/pipeline/registry.py`로 실행
- 단일 파이프라인 재사용: CLI(`run_deck`)와 GUI(`run_simulation`)가 동일한 deck payload 기반 서비스 실행 경로 사용
- GUI 분할: `proc2d/app/gui/forms.py`, `proc2d/app/gui/tabs.py`, `proc2d/app/gui/compare.py`, `proc2d/app/gui/session.py`
- 설정 모델: `proc2d/config/gui_models.py`, `proc2d/config/deck_models.py`, `proc2d/config/parser.py`
- 출력 모듈 분리: `proc2d/export/*` (writer/manager), `proc2d/io.py`는 하위호환 facade
- 도메인/물리/분석 계층: `proc2d/domain/*`, `proc2d/physics/*`, `proc2d/analysis/*`
- step 실행 분해: `proc2d/pipeline/engine.py`, `proc2d/pipeline/context.py`, `proc2d/pipeline/steps/*`

---

## Deck 스키마

기본 구조:

```yaml
domain:
  Lx_um: 2.0
  Ly_um: 0.5
  Nx: 401
  Ny: 201
background_doping_cm3: 1.0e15

steps:
  - type: mask
    openings_um:
      - [0.8, 1.2]
    sigma_lat_um: 0.03

  - type: oxidation
    model: deal_grove
    time_s: 5.0
    A_um: 0.1
    B_um2_s: 0.01
    gamma: 2.27
    apply_on: all
    mask_weighting: binary
    open_threshold: 0.5
    consume_dopants: true
    update_materials: true

  - type: implant
    dose_cm2: 1.0e13
    Rp_um: 0.05
    dRp_um: 0.02

  - type: anneal
    # Option A: constant diffusivity
    D_cm2_s: 1.0e-14

    # Option B: Arrhenius (단일 온도 또는 schedule)
    # diffusivity:
    #   model: arrhenius
    #   D0_cm2_s: 1.0e-3
    #   Ea_eV: 3.5
    #   T_C: 1000.0
    #   schedule:
    #     - t_s: 3.0
    #       T_C: 900.0
    #     - t_s: 2.0
    #       T_C: 1000.0

    total_t_s: 10.0
    dt_s: 0.5
    oxide:
      D_scale: 0.0
    cap_eps_um: 0.001
    cap_model: hard
    # cap_len_um: 0.01  # cap_model=exp일 때 사용
    top_bc:
      open:
        type: robin
        h_cm_s: 1.0e-5
        Ceq_cm3: 0.0
      blocked:
        type: neumann

  - type: analyze
    iso_area:
      threshold_cm3: 1.0e17
      method: tri_linear
    junctions:
      - x_um: 1.0
        threshold_cm3: 1.0e17
    laterals:
      - y_um: 0.05
        threshold_cm3: 1.0e17
    sheet_dose:
      save_csv: true
    save:
      json: true
      csv: true

  - type: export
    outdir: outputs/run1
    formats: [npy, csv, png, vtk]
    linecuts:
      - kind: vertical
        x_um: 1.0
      - kind: horizontal
        y_um: 0.05
    plot:
      log10: true
      vmin: 1.0e14
      vmax: 1.0e20
    extra:
      tox_csv: true
      tox_png: true
```

### `anneal.record` (선택)

```yaml
record:
  enable: true
  every_s: 0.5
  save_csv: true
  save_png: true
```

기록 필드:

- `time_s`
- `mass` (cm^-1)
- `peak_cm3`
- `flux_out` (Robin top BC일 때 의미)
- `residual` (Robin mass balance check)

출력:

- `history.csv`
- `history.png`

### `oxidation` step (선택)

- `model`: 현재 `deal_grove` 지원
- `time_s`, `A_um`, `B_um2_s`: Deal-Grove 파라미터
- `gamma`: Si->SiO2 체적 팽창계수 (기본 2.27)
- `apply_on`: `all | open | blocked`
- `mask_weighting`: `binary | time_scale` (기본 `binary`)
- `open_threshold`: binary 모드 open 판정 임계값
- `consume_dopants`: 산화막 영역 도펀트 0 처리 여부
- `update_materials`: material map 갱신 여부

결과 필드:

- `tox_um(x)`
- `materials` (`0=Si`, `1=SiO2`)

---

## Analyze step 출력

`analyze` step를 넣으면 아래 파일이 생성됩니다.

- `metrics.json`
- `metrics.csv`
- `sheet_dose_vs_x.csv` (옵션)

계산 지표 예:

- total mass
- peak concentration and location
- junction depth at requested x/threshold
- lateral extents at requested y/threshold
- sheet dose summary

---

## VTK 출력 (ParaView)

`export.formats`에 `vtk`를 넣으면 legacy VTK ASCII를 생성합니다.

- `C.vtk`
- `C_log10.vtk` (`plot.log10: true`일 때)
- `material.vtk` (`materials`가 있을 때)

ParaView 사용:

1. `File -> Open -> outputs/run2/C.vtk`
2. `Apply`
3. `Coloring`에서 `doping` 선택

---

## GUI 기능

GUI는 아래를 지원합니다.

- 파라미터 입력 + Run Simulation
- 탭:
  - `Map`: 농도 맵 + mask open overlay
  - `Linecuts`: vertical/horizontal + linear/log10
  - `Metrics`: JSON 표시 + 핵심 값
  - `History`: mass/flux/residual 그래프
  - `Compare`: 최근 2회 실행 before/after 비교
  - `Artifacts`: 파일 목록 및 다운로드
- 추가 토글:
  - oxidation on/off + Deal-Grove 파라미터
  - Arrhenius diffusivity on/off + D0/Ea/T
  - oxide D_scale, cap_eps_um
  - Compute metrics
  - Record anneal history
  - Export VTK
  - Export tox CSV/PNG
  - Download all outputs as ZIP
  - Store full C in session (정밀 비교용)

---

## 예제 Deck

- 기본: `examples/deck_basic.yaml`
- 분석/히스토리/VTK 포함: `examples/deck_analysis_history_vtk.yaml`
- P2 산화+Arrhenius: `examples/deck_oxidation_implant_anneal.yaml`
- Arrhenius schedule 예제: `examples/deck_schedule_anneal.yaml`

실행:

```bash
python3 -m proc2d run examples/deck_analysis_history_vtk.yaml --out outputs/run2
python3 -m proc2d run examples/deck_oxidation_implant_anneal.yaml --out outputs/run_p2
```

---

## 테스트

```bash
python3 -m pytest
```

계층 실행 예시:

```bash
python3 -m pytest -m "unit"
python3 -m pytest -m "not integration and not adapter and not slow"
```

포함 테스트:

- mass conservation
- symmetry
- 1D limit
- metrics 계산
- VTK writer
- anneal history 기록
- Deal-Grove oxidation / surface shift / implant depth shift
- Arrhenius 온도 증가 시 D 증가 경향
- deck/GUI 공통 서비스 parity + artifact manifest 회귀

타입 체크:

```bash
python3 -m mypy
```

---

## 수식/단위

입력 길이는 um, 내부 계산은 cm 단위를 사용합니다.

implant depth profile:

`g(y) = dose / (sqrt(2*pi)*dRp) * exp(-0.5*((y - Rp)/dRp)^2)`

diffusion:

`∂C/∂t = ∇·(D ∇C)`

implicit time integration:

`(I - dt*A) C_{n+1} = C_n + dt*b`

top Robin BC:

`-D ∂C/∂n = h (C - Ceq)`

---

## 모델 가정/한계

- 단일 species
- implant separable 근사
- 농도 의존 D, 다종 coupling, 활성화/클러스터링 등은 미포함
- 고급 물리(활성화, clustering, 다중종 coupling) 미포함
