# proc2d

반도체 공정 2D 단면(Process 2D cross-section) 시뮬레이터 MVP입니다.

이 README는 기존 `PROCESS_2D_ARCHITECTURE_GUIDE` 내용을 통합한 단일 문서입니다.  
다음 세션에서 이 문서 하나만 읽어도 구조/공식/실행 방법을 빠르게 복구할 수 있도록 작성했습니다.

## 1) 목표와 MVP 범위

- 2D 단면에서 `마스크 -> 이온주입 -> 확산(어닐)` 파이프라인 실행
- 좌표계: `x`(lateral), `y`(depth), `y=0`이 웨이퍼 표면
- 상태 변수: 단일 도핑 농도장 `C(y, x)` [cm^-3]
- 구현 범위:
  - 1D 마스크(openings + lateral Gaussian smoothing)
  - 2D 이온주입(separable Gaussian)
  - 2D 확산(implicit Backward Euler + sparse solver)
  - 상부 mixed BC(open=Robin, blocked=Neumann)
  - YAML deck 기반 실행 + `npy/csv/png` 출력

## 2) 설치

```bash
pip install -e ".[dev]"
```

Debian/Ubuntu에서 PEP 668(externally-managed-environment) 오류가 나면:

```bash
python3 -m pip install --user --break-system-packages -e ".[dev]"
```

GUI까지 사용하려면:

```bash
python3 -m pip install --user --break-system-packages -e ".[gui]"
```

의존성: `numpy`, `scipy`, `PyYAML`, `matplotlib`, `pytest`

## 3) 실행

```bash
python3 -m proc2d run examples/deck_basic.yaml --out outputs/run1
```

또는 콘솔 스크립트:

```bash
proc2d run examples/deck_basic.yaml --out outputs/run1
```

`proc2d`가 PATH에 없으면 모듈 방식(`python3 -m proc2d ...`)을 사용하세요.

`C.png`는 "방금 실행한 시뮬레이션"의 최종 농도 맵입니다.  
아무 파라미터를 바꾸지 않고 실행하면 `examples/deck_basic.yaml`과 동일한 기본값 결과를 보게 됩니다.

## 3-1) GUI 실행

아래 명령으로 GUI를 띄워서 파라미터를 바꾸고, 버튼으로 실행한 뒤 같은 화면에서 결과 맵/라인컷을 볼 수 있습니다.

```bash
proc2d-gui
```

`proc2d-gui`가 PATH에 없으면:

```bash
python3 -m streamlit run proc2d/gui_script.py
```

GUI 기능:

- 예제 deck 기본값 자동 로드
- 파라미터 조정 (domain/mask/implant/anneal/BC/export)
- `Run Simulation` 버튼으로 즉시 실행
- 같은 화면에서 `C.png` 표시
- x/y 위치를 바꿔 linecut 그래프 확인
- linecut CSV 다운로드

## 4) YAML deck 스키마

기본 예시:

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

  - type: implant
    dose_cm2: 1.0e13
    Rp_um: 0.05
    dRp_um: 0.02

  - type: anneal
    D_cm2_s: 1.0e-14
    total_t_s: 10.0
    dt_s: 0.5
    top_bc:
      open:
        type: robin
        h_cm_s: 1.0e-5
        Ceq_cm3: 0.0
      blocked:
        type: neumann

  - type: export
    outdir: outputs/run1
    formats: [npy, csv, png]
    linecuts:
      - kind: vertical
        x_um: 1.0
      - kind: horizontal
        y_um: 0.05
    plot:
      log10: true
      vmin: 1.0e14
      vmax: 1.0e20
```

### domain
- `Lx_um`, `Ly_um`: 시뮬레이션 영역 크기 [um]
- `Nx`, `Ny`: 격자 포인트 수 (`x_um = linspace(0, Lx_um, Nx)`, `y_um = linspace(0, Ly_um, Ny)`)
- `background_doping_cm3`: 초기 배경 농도 [cm^-3]

### steps
- `mask`
  - `openings_um`: open 구간 리스트 (`[[x0, x1], ...]`)
  - `sigma_lat_um`: lateral smoothing sigma [um], `0`이면 binary mask
- `implant`
  - `dose_cm2`: 주입 dose [cm^-2]
  - `Rp_um`, `dRp_um`: 깊이 Gaussian 파라미터 [um]
  - 적용식: `C(y,x) += g(y) * mask_eff(x)`
- `anneal`
  - `D_cm2_s`: 확산계수 [cm^2/s]
  - `total_t_s`, `dt_s`: 총 시간, 시간 스텝 [s]
  - `top_bc.open.type`: `robin | neumann | dirichlet`
    - Robin: `-D dC/dn = h (C - Ceq)`
    - 입력: `h_cm_s`, `Ceq_cm3`
  - `top_bc.blocked.type`: MVP에서 `neumann`만 지원
  - `mask` step이 없으면 전체 open으로 처리
- `export`
  - `outdir`: 출력 폴더
  - `formats`: `npy`, `csv`, `png`
  - `linecuts`: vertical/horizontal 라인컷
  - `plot.log10`: 로그 플롯 여부

## 5) 출력 파일

- `outputs/run1/C.npy`: 최종 농도 필드 (`shape=[Ny, Nx]`)
- `outputs/run1/C.png`: 2D heatmap (`plot.log10=true`면 `log10(C)`)
- `outputs/run1/linecut_*.csv`: 라인컷 데이터

## 6) 테스트

```bash
python3 -m pytest
```

포함 테스트:

1. Mass conservation: 전 경계 Neumann에서 총량 보존
2. Symmetry: 중앙 대칭 마스크 조건에서 x 대칭성 유지
3. 1D limit: 전체 open + x 균일 조건에서 x 방향 편차 최소

## 7) 코드 구조(아키텍처)

- `proc2d/units.py`
  - um<->cm 변환, 양수/비음수 검증
- `proc2d/grid.py`
  - `Grid2D`(격자 생성/간격/인덱스 유틸)
- `proc2d/mask.py`
  - openings 기반 1D mask 생성 + Gaussian smoothing
- `proc2d/implant.py`
  - 깊이 Gaussian `g(y)` 계산, `dC(y,x)=g(y)*mask_eff(x)` 적용
- `proc2d/diffusion.py`
  - 확산 연산자 조립, implicit step, top BC(Robin/Neumann/Dirichlet-demo)
- `proc2d/io.py`
  - `npy/csv/png` export
- `proc2d/deck.py`
  - YAML 파싱 + step 실행 엔진 + 상태 관리
- `proc2d/cli.py`
  - `proc2d run ...` CLI

## 8) 실행 파이프라인

1. `domain` 파싱 -> `Grid2D` 생성 -> `C`를 background로 초기화
2. `steps` 순차 실행
   - `mask`: `mask_eff(x)` 저장
   - `implant`: `C += dC`
   - `anneal`: implicit 확산 시간적분
   - `export`: 파일 출력
3. 최종 상태 반환

동작 규칙:

- `mask`가 없으면 implant/anneal은 전체 open으로 처리
- `anneal.top_bc`는 상부 경계에만 적용
- `--out` 오버라이드는 export 출력 경로를 강제로 지정

## 9) 반도체 모델/공식 정리

### 9.1 단위 규칙

- 입력 길이: um
- 내부 계산: cm 기준으로 통일
  - `D` [cm^2/s]
  - dose [cm^-2]
  - `C` [cm^-3]

### 9.2 마스크와 lateral straggle 근사

- binary mask를 1D Gaussian convolution으로 스무딩
- `sigma_lat_um=0`이면 binary 그대로
- 결과는 `[0,1]` clip

### 9.3 이온주입(깊이 Gaussian)

`g(y) = dose / (sqrt(2*pi)*dRp) * exp(-0.5*((y - Rp)/dRp)^2)`

- `dose` [cm^-2], `y/Rp/dRp` [cm], `g(y)` [cm^-3]
- 연속계에서 `∫ g(y) dy = dose`

2D separable 적용:

`C(y,x) += g(y) * mask_eff(x)`

### 9.4 확산 PDE

`∂C/∂t = ∇·(D ∇C)`

MVP에서는 상수 `D`를 사용.

시간 적분(Backward Euler):

`(I - dt*A) C_{n+1} = C_n + dt*b`

- `A`: 공간 이산 연산자
- `b`: Robin/Dirichlet에서 생기는 상수항

### 9.5 상부 경계조건 (mask 기반)

- 좌/우/하부: Neumann(0-flux)
- 상부(y=0):
  - open 영역: Robin
  - blocked 영역: Neumann

Robin:

`-D ∂C/∂n = h (C - Ceq)`

- `h` [cm/s], `Ceq` [cm^-3]
- `Ceq=0`이면 out-diffusion sink

이산화 반영(상부 셀):

- 대각항: `-h/dy`
- 상수항: `+h*Ceq/dy`
- mask가 부분 open이면 `h_eff = mask_eff * h` 사용

## 10) 수치 구현 디테일

- 배열 shape: `(Ny, Nx)`
- flatten 인덱스: `k = j*Nx + i` (row-major)
- 공간 이산화: 5-point stencil
- 선형계:
  - matrix: `scipy.sparse.csr_matrix`
  - solve: `spsolve`
  - 반복 step에서 고정 `dt`는 `splu` 재사용

## 11) 모델 가정/한계

- 단일 species 농도장 `C(y,x)`만 사용
- implant는 separable 근사(`g(y)*mask(x)`), tilt implant 미지원
- 확산계수 `D`는 상수(농도/온도 의존성 미포함)
- 고급 물리(활성화, clustering, 전기장 결합)는 MVP 범위 밖

## 12) 확장 로드맵

- 물리 확장:
  - `D(C,T)` 의존성
  - multi-species (B, P, As)
  - 활성화/비활성화, clustering
  - tilt implant
- 수치 확장:
  - variable D + face harmonic mean
  - iterative solver + preconditioner
  - adaptive dt
- 입출력 확장:
  - step별 snapshot
  - HDF5
  - 추가 라인컷 포맷(JSON/Parquet)

## 13) 다음 세션 빠른 재현 절차

```bash
python3 -m pip install --user --break-system-packages -e ".[dev]"
python3 -m pytest
python3 -m proc2d run examples/deck_basic.yaml --out outputs/run1
proc2d-gui
```
