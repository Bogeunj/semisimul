# proc2d 모듈화 계획

작성일: 2026-02-26  
분석 경로: `/home/bogeun/projects/semisimul_b`

## 1) 분석 범위와 결론

요청대로 저장소의 구성 요소를 전체 확인했다. 핵심 결론은 다음과 같다.

- 물리/수치 계산 모듈(`mask`, `implant`, `oxidation`, `diffusion`, `metrics`)은 이미 파일 단위 분리가 잘 되어 있다.
- 반면 실행 오케스트레이션이 `proc2d/deck.py`와 `proc2d/gui_app.py`에 이중화되어 있어, 기능 확장 시 동일 변경을 2군데 이상 반영해야 한다.
- 특히 `proc2d/gui_app.py`가 1333라인 단일 파일로 커져 UI/시뮬레이션/파일출력/세션관리/비교 기능이 혼합되어 유지보수 리스크가 높다.
- 따라서 모듈화의 1순위는 **공통 Simulation Service 추출 + Step Registry 기반 파이프라인화 + GUI 분할**이다.

---

## 2) 현재 프로젝트 전체 구성

### 2.1 루트 구조

```text
.
├── .github/workflows/ci.yml
├── examples/
│   ├── deck_basic.yaml
│   ├── deck_analysis_history_vtk.yaml
│   ├── deck_oxidation_implant_anneal.yaml
│   └── deck_schedule_anneal.yaml
├── proc2d/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── deck.py
│   ├── diffusion.py
│   ├── grid.py
│   ├── gui.py
│   ├── gui_app.py
│   ├── gui_script.py
│   ├── implant.py
│   ├── io.py
│   ├── mask.py
│   ├── metrics.py
│   ├── oxidation.py
│   └── units.py
├── scripts/
│   ├── setup_user_install.sh
│   └── setup_venv.sh
├── tests/
│   ├── test_1d_limit.py
│   ├── test_history.py
│   ├── test_mass_conservation.py
│   ├── test_metrics.py
│   ├── test_oxidation_p2.py
│   ├── test_symmetry.py
│   └── test_vtk_writer.py
├── pyproject.toml
├── README.md
├── PROJECT_STATUS_REPORT.md
└── get-pip.py
```

### 2.2 패키징/실행 인터페이스

- 패키지명/버전: `proc2d` / `0.1.0`
- Python: `>=3.10`
- 핵심 의존성: `numpy`, `scipy`, `PyYAML`, `matplotlib`
- 선택 의존성: `pytest`(dev), `streamlit`(gui)
- 엔트리포인트:
  - CLI: `proc2d = proc2d.cli:main`
  - GUI: `proc2d-gui = proc2d.gui:main`

### 2.3 모듈별 책임(코드 규모 포함)

| 모듈 | 라인 수 | 책임 | 비고 |
|---|---:|---|---|
| `proc2d/units.py` | 47 | 단위 변환, 공통 scalar/array 유효성 검사 | 공통 유틸 |
| `proc2d/grid.py` | 89 | `Grid2D` 도메인 모델(격자 생성/인덱싱) | core 데이터 모델 |
| `proc2d/mask.py` | 88 | mask 생성/가우시안 smoothing | 전처리 물리 모듈 |
| `proc2d/implant.py` | 71 | Gaussian implant 계산(`tox` 고려) | 도핑 소스 모델 |
| `proc2d/oxidation.py` | 144 | Deal-Grove 산화/표면 이동/material map | P2 핵심 확장 |
| `proc2d/diffusion.py` | 393 | 확산 연산자 조립/implicit solver/BC/history | 수치 핵심 엔진 |
| `proc2d/metrics.py` | 213 | 정량 분석(peak, junction, lateral, sheet dose) | analyze step 기반 |
| `proc2d/io.py` | 425 | NPY/CSV/PNG/VTK/history/metrics 저장 | 출력 책임이 큼 |
| `proc2d/deck.py` | 587 | YAML 파싱 + step 실행 엔진 + 상태관리 | 오케스트레이션 허브 |
| `proc2d/cli.py` | 47 | CLI argument 파싱/실행 | 얇은 어댑터 |
| `proc2d/gui.py` | 22 | streamlit launcher | 얇은 어댑터 |
| `proc2d/gui_script.py` | 6 | streamlit script 진입점 | 래퍼 |
| `proc2d/gui_app.py` | 1333 | GUI 화면 + 시뮬 실행 + 비교/다운로드 | 과대 단일 모듈 |

### 2.4 내부 의존 관계

현재 내부 의존은 대략 아래와 같다.

```text
deck -> diffusion, grid, implant, io, mask, metrics, oxidation, units
gui_app -> diffusion, grid, implant, io, mask, metrics, oxidation
diffusion/grid/implant/mask/oxidation -> units/grid 기반
cli -> deck
```

의존 방향은 기본적으로 좋지만, `deck`와 `gui_app`가 같은 하위 모듈을 각각 조합하면서 실행 로직이 중복된다.

---

## 3) 실제 실행 구조 분석

### 3.1 CLI 실행 흐름

1. `python -m proc2d run <deck.yaml>`
2. `proc2d/cli.py`에서 `run_deck` 호출
3. `proc2d/deck.py`에서
   - deck 로드/검증
   - `SimulationState` 초기화
   - `steps` 순차 실행 (`mask -> oxidation -> implant -> anneal -> analyze -> export`)
4. `state.exports`에 산출물 경로 누적 후 종료

### 3.2 GUI 실행 흐름

1. `proc2d-gui` 또는 `streamlit run proc2d/gui_script.py`
2. `proc2d/gui_app.py`의 `run_gui()`에서 폼 입력 수집
3. 같은 파일의 `run_simulation(params)`에서 직접 시뮬레이션 수행
   - grid 생성
   - mask/oxidation/implant/anneal
   - export/metrics/history 저장
4. 탭(Map/Linecuts/Metrics/History/Compare/Artifacts) 렌더링

핵심 문제는 3.2의 계산 파이프라인이 3.1(`deck.py`)과 별도로 구현되어 있다는 점이다.

---

## 4) 구조상 강점과 한계

### 4.1 강점

- 물리 연산(implant/oxidation/diffusion/metrics)이 모듈로 나뉘어 있어 수치 로직 자체는 재사용 가능하다.
- Deck 기반 step 파이프라인 컨셉이 이미 있어 확장 여지가 크다.
- 테스트가 물리 성질(질량보존/대칭/1D limit)과 기능(history/vtk/oxidation)을 포함한다.

### 4.2 한계(모듈화 대상)

1. **오케스트레이션 중복**
   - `deck.py`와 `gui_app.py`가 각각 step 조립, Arrhenius 처리, export/metrics/history 저장을 수행.
   - 동일 기능 개선 시 다중 파일 수정 필요.

2. **GUI 단일 파일 과대화**
   - `gui_app.py`가 UI + 도메인 실행 + 비교 + 세션 + 파일 다운로드를 모두 담당.
   - 테스트 작성, 코드 탐색, 병렬 개발이 어렵다.

3. **구성 모델이 dict 중심**
   - deck/GUI params가 대부분 `dict[str, Any]`로 흐른다.
   - 스키마 변경 시 컴파일 단계 보호가 약하고 런타임 에러를 유발하기 쉽다.

4. **step 확장성 제한**
   - `deck.py`의 `if/elif` 분기에 step 추가를 직접 연결해야 한다.
   - 플러그인 형태의 step registry가 없다.

5. **I/O 책임 혼합**
   - `io.py`에 필드 저장, plot, VTK, metrics flatten, history plot 등 이질 책임이 섞여 있다.

6. **테스트 편중**
   - 핵심 수치 테스트는 있으나, deck validation 경계조건/GUI 서비스 레벨 테스트는 상대적으로 약하다.

---

## 5) 모듈화 목표(향후 개발 기준)

모듈화는 "파일 쪼개기"가 아니라 **변경의 방향을 분리**하는 것이 목표다.

- 단일 실행 엔진: CLI/GUI가 동일한 서비스 계층을 호출하도록 통합
- 스키마 일원화: deck/GUI 모두 같은 typed config 모델 사용
- step 확장 용이성: step registry로 신규 step 추가 시 영향 범위 최소화
- UI 독립성: Streamlit UI는 view/controller 역할만, 물리 계산은 service로 분리
- 출력 분리: export writer를 형식별 모듈로 분해
- 테스트 구조화: physics / pipeline / adapters(UI/CLI) 계층별 테스트 분리

---

## 6) 제안 타깃 아키텍처

아래는 현재 코드와 충돌이 적은 현실적인 목표 구조다.

```text
proc2d/
  app/
    cli.py
    gui/
      app.py
      forms.py
      tabs.py
      compare.py
      session.py

  config/
    deck_models.py
    gui_models.py
    parser.py
    validators.py

  domain/
    grid.py
    state.py
    units.py
    constants.py

  physics/
    mask.py
    implant.py
    oxidation.py
    diffusion.py

  analysis/
    metrics.py
    reports.py

  pipeline/
    engine.py
    context.py
    step_base.py
    registry.py
    steps/
      mask_step.py
      oxidation_step.py
      implant_step.py
      anneal_step.py
      analyze_step.py
      export_step.py

  export/
    manager.py
    npy_writer.py
    csv_writer.py
    png_writer.py
    vtk_writer.py
    history_writer.py
    metrics_writer.py

  services/
    simulation_service.py
    compare_service.py
```

핵심 포인트:

- `services/simulation_service.py`가 "유일한 실행 진입점"이 된다.
- `deck.py`, `gui_app.py`는 서비스 호출/입출력 변환만 담당한다.
- 기존 파일은 즉시 삭제하지 않고 thin wrapper로 유지해 하위호환을 지킨다.

---

## 7) 단계별 모듈화 실행 계획

### Phase 0. 안전장치 확보 (선행)

- 현재 동작 고정용 회귀 테스트 추가
  - 동일 입력(deck vs gui params)에서 핵심 출력 동등성 확인
  - 최소 비교 대상: `C` 통계, `metrics`, `history` 길이/단조성
- 결과물 manifest(파일명 리스트) 검증 테스트 추가

완료 기준:

- 리팩터링 전후 테스트가 동일 통과
- 기능 회귀 없이 구조 변경 가능 상태 확보

### Phase 1. Config 스키마 분리

- `dict[str, Any]` 입력을 typed model로 전환
  - `DomainConfig`, `MaskStepConfig`, `AnnealConfig` 등
- deck parser와 GUI params 변환기를 `config/` 계층으로 이동

완료 기준:

- `deck.py` 내부의 `_to_float/_to_int/_required`류 유효성 함수 의존 축소
- 잘못된 입력 에러 메시지 일관화

### Phase 2. Pipeline 엔진/Step Registry 도입

- `pipeline.engine.run(steps, context)` 도입
- step별 실행함수 분리(`pipeline/steps/*.py`)
- registry 매핑으로 step 추가/삭제를 분기문 없이 처리

완료 기준:

- `deck.py`는 load + service 호출 + 결과 반환 중심으로 축소
- 신규 step 추가 시 registry + step file만 수정

### Phase 3. 공통 Simulation Service 통합 (최우선)

- `services/simulation_service.py`에서 단일 실행 흐름 제공
  - grid 초기화
  - mask/oxidation/implant/anneal/analyze/export
  - 결과 dataclass 반환
- `deck.py`와 `gui_app.py` 모두 동일 서비스 사용

완료 기준:

- Arrhenius/oxidation/export/metrics/history 로직 중복 제거
- CLI와 GUI 결과 일치성 검증 테스트 통과

### Phase 4. GUI 모듈 분할

- `gui_app.py`를 기능 단위 파일로 분리
  - forms 입력
  - 탭 렌더러(Map/Linecuts/Metrics/History/Compare/Artifacts)
  - session 관리
- UI에서 파일 I/O 직접 호출을 줄이고 service 결과만 표시

완료 기준:

- `gui_app.py` 대폭 축소(엔트리/조립 수준)
- 탭 단위 테스트 또는 최소 렌더 smoke 테스트 가능

### Phase 5. Export/Analysis 모듈 정리

- `io.py`를 writer 단위로 분해
- 포맷별 writer 인터페이스 정의(`write(field, context) -> Path`)
- analyze report 생성과 저장을 분리

완료 기준:

- 출력 포맷 추가(예: npz/hdf5) 시 기존 코드 수정 최소화
- `io.py` 단일 거대 모듈 제거

### Phase 6. 품질/운영 보강

- 테스트 계층 분리
  - unit(physics)
  - integration(pipeline)
  - adapter(cli/gui)
- typing 강화(mypy/pyright 중 택1)
- CI에 lint/type check 선택적 추가

완료 기준:

- 기능 추가 PR에서 영향 범위를 모듈 단위로 명확히 제한 가능

---

## 8) 기존 파일 -> 목표 모듈 매핑 제안

| 현재 | 목표(예시) | 전략 |
|---|---|---|
| `proc2d/deck.py` | `config/*`, `pipeline/*`, `services/simulation_service.py` | 단계 분해 후 wrapper 유지 |
| `proc2d/gui_app.py` | `app/gui/*` + `services/simulation_service.py` | UI/실행 분리, 탭 분할 |
| `proc2d/io.py` | `export/*` + `analysis/reports.py` | writer 단위 분리 |
| `proc2d/diffusion.py` | `physics/diffusion.py` (필요시 BC 분리) | 기능 유지, 경계조건 구성만 분리 가능 |
| `proc2d/metrics.py` | `analysis/metrics.py` | 현 API 유지 후 경로만 이관 |
| `proc2d/grid.py`, `units.py` | `domain/*` | core 타입 계층으로 승격 |

---

## 9) 리스크와 대응

1. 리팩터링 중 수치 결과 미세 변경 가능
   - 대응: 상대오차 기반 회귀 테스트, 골든 케이스 비교

2. GUI 기능(특히 Compare/다운로드) 깨질 위험
   - 대응: 탭별 smoke 시나리오 + 서비스 반환 스키마 고정

3. 과도한 일괄 개편으로 개발 속도 저하
   - 대응: Phase별 PR 분리(작게, 자주), wrapper 호환 유지

---

## 10) 바로 착수 가능한 우선순위 (권장)

1. `simulation_service` 신설 후 `deck.py` 먼저 연결
2. 동일 서비스에 GUI `run_simulation` 연결하여 중복 로직 제거
3. `gui_app.py`를 `forms.py`/`tabs.py`/`compare.py`로 1차 분리
4. 마지막으로 `io.py` writer 분해

이 순서가 가장 리스크가 낮고, 체감 유지보수성을 빠르게 개선한다.
