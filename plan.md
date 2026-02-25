# proc2d 개선 계획 (plan.md)

작성일: 2026-02-25  
대상: `proc2d` (deck 엔진 + CLI + Streamlit GUI + metrics/oxidation/diffusion/io)

---

## 0. 배경과 문제 정의

현재 `proc2d`는 **deck(YAML) 기반 엔진(`deck.py`)**과 **GUI 파이프라인(`gui*.py`)**이 유사한 시뮬레이션 흐름을 각각 구현하고 있다.

리서치에서 확인된 주요 개선 포인트는 다음 4가지다.

1) **GUI 계산 경로를 deck 엔진 호출 방식으로 통합**
   - 로직 중복 제거, 기능 편차(예: schedule) 해소, 유지보수성 향상

2) **analyze/metrics 확장**
   - 특히 `iso_contour_area`가 cell-count 근사라 정확도가 제한됨 -> contour 기반(보간 포함) 옵션 필요

3) **BC/산화 coupling 고도화**
   - oxidation에서 `mask_eff>0.5` 이진화로 fractional edge 효과가 반영되지 않음 -> mask fractional 가중 적용 옵션 필요
   - diffusion top BC에서도 tox/oxide에 따른 "effective open" 모델 개선 여지

4) **환경 재현성 강화**
   - 현재 환경에서 `numpy/pytest` 부재로 실행/테스트가 즉시 불가 -> 의존성/설치 가이드 + self-check 필요

이 문서는 위 4개 포인트를 "기능 유지 + 점진적 리팩터링 + 테스트 기반 안정화" 원칙으로 구현하기 위한 실행 계획이다.

---

## 1. 목표와 비목표

### 1.1 목표

- **단일 엔진(single source of truth)**: 시뮬레이션 코어 실행 경로를 1개로 만들고(Engine), CLI/Deck/GUI는 이 엔진을 호출한다.
- **기존 deck 스키마/CLI 사용성 유지**: 기존 YAML deck과 `python -m proc2d run ...` 동작을 깨지 않는다.
- **GUI-Deck 기능 일관성 확보**: schedule/analysis 구성 등에서 "가능하면 동일 기능"을 제공한다.
- **정확도 옵션 추가(기본값 유지)**: `iso_contour_area` 등은 새 방법을 "옵션"으로 추가하고, 기본값은 기존과 동일하게 유지한다.
- **재현 가능한 실행 환경 제공**: 최소 설치/개발 설치/GUI 설치를 분리하고, `selfcheck`로 빠르게 검증한다.

### 1.2 비목표(이번 범위에서 하지 않음)

- 완전한 TCAD급 공정 모델(etch/deposition 상세 형상 진화, 스트레스/전하/전기적 특성까지의 full stack)
- 3D 확장
- 물질/도펀트 다중종 확산 + 반응 모델(현 구조는 1종 농도장 중심)

---

## 2. 성공 기준(Definition of Done)

아래 조건을 만족하면 "개선 완료"로 본다.

- **기능 동일성**
  - 기존 예제 deck(`examples/*.yaml`)이 모두 실행되고(의존성 충족 시), **기존 대비 metrics가 허용 오차 내에서 동일**하다.
- **구조 통합**
  - GUI는 더 이상 자체 파이프라인을 "따로" 갖지 않고, 엔진 호출로 실행된다(중복 제거).
- **옵션 추가**
  - `iso_contour_area`에 새 계산법이 추가되며, 기본값은 기존과 동일하다.
  - oxidation mask fractional 가중 옵션이 추가되며, 기본값은 기존과 동일하다.
- **재현성**
  - `pip install -e .[dev]` 후 `pytest`가 동작한다.
  - `python -m proc2d selfcheck`가 설치/핵심 기능을 빠르게 검증한다.

---

## 3. 우선순위 및 단계(Milestones)

- **M0 (P0)**: 환경 재현성/설치/CI 기반 정리 -> "테스트가 도는 상태" 만들기
- **M1 (P0)**: 엔진 계층 도입 + deck 실행 경로를 엔진으로 이전(외부 동작 동일)
- **M2 (P0)**: GUI를 엔진 기반으로 전환(중복 로직 제거) + GUI/Deck parity 테스트 추가
- **M3 (P1)**: `iso_contour_area` 정확도 옵션 추가(보간 기반) + 테스트
- **M4 (P1)**: oxidation mask fractional 가중 옵션 + diffusion top BC "effective open" 고도화 옵션 + 테스트

> 참고: M0->M2는 "유지보수/기능일관성"에 직결되어 최우선(P0)로 진행한다.  
> M3/M4는 물리/수치 모델 개선(P1)이며 기본값은 유지한다.

---

## 4. 설계 방향(핵심 아키텍처)

### 4.1 "Engine" 계층 도입: 실행 로직을 1곳으로 모으기

#### 목표

- `deck.py`의 "실행"과 `gui.py`의 "실행"을 **공통 엔진 호출로 통합**
- deck은 "파싱/검증/정규화"에 집중, 실행은 엔진으로 위임
- GUI는 "UI -> deck dict 생성" 후 엔진 호출

#### 제안 모듈 구조(예시)

- `proc2d/engine.py` (신규): 공통 실행기
- `proc2d/deck.py`: YAML 로딩 + 검증 + normalize, 그리고 `engine.run()` 호출
- `proc2d/gui*.py`: UI 파라미터 -> deck dict 생성(`build_deck_from_ui`) -> `engine.run()`

#### 엔진 API 초안(제안)

```python
# proc2d/engine.py (신규)

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class RunResult:
    state: "SimulationState"
    metrics: dict
    history: dict | None
    exports: dict  # 생성된 artifact 메타(경로/바이트/키 등)


class ArtifactSink(Protocol):
    def save_bytes(self, relpath: str, data: bytes) -> None: ...
    def save_text(self, relpath: str, text: str) -> None: ...
    def save_array(self, relpath: str, arr: "np.ndarray") -> None: ...
    def make_subdir(self, relpath: str) -> "ArtifactSink": ...


class FileSystemSink(ArtifactSink):
    ...


class MemorySink(ArtifactSink):
    ...


def run_deck_mapping(
    deck: dict,
    *,
    base_dir: Path | None,
    out_override: Path | None,
    sink: ArtifactSink | None,
    hooks: "RunHooks | None" = None,
) -> RunResult:
    ...
```

핵심은 I/O를 분리하는 것이다.

- CLI/Deck: `FileSystemSink`
- GUI: `MemorySink` 또는 "임시 폴더 FileSystemSink + 메모리 로딩" 중 선택 가능

장점:

- deck과 GUI가 같은 step executor를 쓰므로 기능 편차 감소
- export/plot이 "저장소(sink)"만 바꿔도 재사용 가능

---

## 5. 작업 계획(Workstreams)

### Workstream A (P0): 환경 재현성 강화 (M0)

#### A1. `pyproject.toml` 의존성 정리

- **필수 의존성(최소 실행)**에 아래를 명시
  - `numpy` (필수)
  - `scipy` (`splu`, `gaussian_filter` 등으로 사실상 필수)
  - `pyyaml` (deck)
  - `matplotlib` (png plot)
- 옵션 의존성으로 분리
  - `dev`: `pytest`, `pytest-cov`, `ruff`(또는 `flake8`), (선택) `mypy`
  - `gui`: `streamlit` (+ GUI에 필요한 것들)

예:

- `pip install -e .` -> CLI 실행 가능
- `pip install -e .[dev]` -> 테스트/개발 가능
- `pip install -e .[gui]` -> GUI 실행 가능
- `pip install -e .[dev,gui]` -> 풀셋

#### A2. "빠른 실패"용 self-check 추가

- CLI 서브커맨드 추가: `python -m proc2d selfcheck`
- selfcheck 항목(빠르게 끝나야 함)
  - import 체크: `numpy/scipy/pyyaml/matplotlib`
  - 최소 grid 생성 + mask 1회 + implant 1회 + anneal 짧게 1~2 step
  - metrics 계산 1회
  - (`FileSystemSink` 선택 시) 임시 디렉터리에 `npy/csv/png` 1개씩 저장 테스트

#### A3. README Quickstart 업데이트 + scripts

- Quickstart 섹션 추가(최소 설치 / dev 설치 / gui 설치)
- `scripts/bootstrap_dev.sh`(선택): venv 생성 + dev 설치 + selfcheck

#### A4. CI workflow 개선

- CI에서 `pip install -e .[dev]` 후 `pytest`
- (선택) GUI smoke는 별도 job로 `pip install -e .[gui]` + import 정도만

완료 조건:

- "새 머신/새 venv"에서도 문서대로 설치하면 selfcheck와 pytest가 재현 가능

### Workstream B (P0): 엔진 계층 도입 + deck 실행 경로 통합 (M1)

#### B1. deck 실행 로직 분리(파싱 vs 실행)

`deck.py` 내부에서 다음을 분리:

- `load_deck(path)`: YAML -> dict
- `validate/normalize(deck, base_dir)`: 타입/필수키/단위/기본값 확정
- `engine.run_deck_mapping(...)`: 실행

핵심 포인트:

- outdir 상대경로 규칙이 "deck 파일 위치 기준"이었으므로,
  normalize 단계에서 `base_dir`를 명시적으로 받아서 항상 기준이 결정되게 한다.
- GUI는 `base_dir=cwd` 또는 임시 폴더를 전달할 수 있어야 한다.

#### B2. step executor를 엔진으로 이동(또는 공통화)

현재 deck 엔진에 step별 실행 규약이 명확하므로,
step 처리 함수들을 `engine.py` 또는 `proc2d/steps/*.py`로 옮겨
GUI/CLI가 동일 함수를 호출하도록 만든다.

#### B3. I/O를 sink 기반으로 간접화

`io.py`의 저장 함수들을 "경로 문자열" 대신 "sink"를 받도록 오버로드/래핑

예:

- `save_metrics(sink, metrics_dict)`
- `save_png(sink, "C.png", fig_or_array, ...)`

이 변경은 단번에 크게 바꾸기보다,

1) 기존 경로 기반 함수 유지 + sink wrapper(경로를 sink로 변환)
2) 내부 구현을 sink 중심으로 점진 이동

전략이 안전하다.

#### B4. 회귀 테스트(golden) 추가

- `examples/*.yaml`을 몇 개 선정해 "golden metric" 테스트를 추가
- `metrics.json`의 주요 키(질량, peak, junction depth 등)를 tolerance 비교
- 수치 solver 특성상 절대 일치가 아닌 상대/절대 오차 허용을 명시

완료 조건:

- CLI/deck 실행 결과가 "외부 관점"에서 이전과 동일
- 테스트가 엔진 기반으로도 안정적으로 통과

### Workstream C (P0): GUI를 엔진 기반으로 통합 (M2)

#### C1. GUI 입력 -> deck mapping 생성 함수 추가

`proc2d/gui.py` 또는 `proc2d/gui_script.py`에
`build_deck_from_ui(params) -> dict` 추가

- 생성되는 dict는 deck YAML 구조와 동일하게 맞춘다.
- `mask/oxidation/implant/anneal/analyze/export` step 키 이름을 일치시킨다.
- GUI 고정 analyze 템플릿은 "preset 함수"로 관리(아래 C3 참고)

#### C2. GUI 실행 경로 변경

기존 `run_simulation` 내부 계산을 제거/축소하고,
`engine.run_deck_mapping(build_deck_from_ui(...), sink=MemorySink(), hooks=...)`로 실행

GUI가 필요로 하는 산출물:

- 최종 `state.C`, `tox_um`, `materials`, `mask_eff`
- `metrics` dict
- `history` dict
- export artifact(다운로드용 zip 등)

=> `RunResult`로 한 번에 반환되도록 엔진을 맞춘다.

#### C3. GUI "분석 preset" 정리

GUI analyze가 고정 템플릿이므로, 엔진과 일관되게:

- `proc2d/presets.py` (신규)로 preset 생성 함수를 만든다.
- 예: `make_default_gui_analyze_step(thresholds=[1e17,1e18], laterals=[...], junctions=[...])`

#### C4. schedule 지원(기능 편차 해소)

deck 엔진은 schedule을 지원하므로, GUI도 다음 중 하나를 제공:

- 간단 모드: 단일 T 모델(현 GUI와 동일)
- 고급 모드: JSON/YAML 텍스트 입력으로 schedule 리스트 입력
- 입력된 schedule을 그대로 deck anneal step에 주입

UI를 복잡하게 만들기 싫으면 "고급 텍스트 입력" 방식이 구현 비용 대비 효과가 크다.

#### C5. GUI/Deck parity 테스트

- 같은 파라미터로 GUI가 생성한 deck dict와 예제 deck을 비교하거나,
- 최소한 동일 조건에서 engine run 결과의 주요 metrics 비교

완료 조건:

- GUI 내부에 "독자적인 시뮬레이션 계산 로직"이 남지 않음(표시/다운로드 로직 제외)
- deck/GUI 기능 차이가 구조적으로 줄어듦

### Workstream D (P1): `iso_contour_area` 정확도 옵션 추가 (M3)

#### D1. 새 면적 계산 방식 추가(기본값 유지)

현재 `iso_contour_area`가 cell-count 근사이므로 아래 옵션을 추가한다.

- 기존: `method="cell_count"` (default, 기존 동작 유지)
- 신규: `method="tri_linear"` (권장)

각 cell을 2개 삼각형으로 분할하고,
삼각형 내부에서 스칼라장을 "선형"으로 가정하여
threshold 이상의 영역을 edge 교차점 보간으로 정확히 계산해 면적을 합산한다.

왜 `tri_linear`인가?

- 외부 의존성 증가 없이(`scikit-image` 불필요) 구현 가능
- cell-count 대비 정확도 크게 향상
- 구현 복잡도는 marching squares보다 낮고, 테스트하기 쉽다

#### D2. API/Deck 연결

analyze step 옵션에 다음 중 하나를 추가:

- `iso_area.method: "cell_count" | "tri_linear"`
- 또는 `iso_contour_area.method` 형태

기존 deck에는 해당 키가 없으므로 default는 `cell_count`

#### D3. 테스트 설계

단위 테스트:

- 상수장: threshold 이하/이상에서 면적 0/전체 정확히
- 선형 램프: 면적이 이론값(예: 정확히 절반)에 가까운지
- 해상도 변화: grid refinement 시 `tri_linear` 오차가 줄어드는지(수렴성 체크)

회귀 테스트:

- 기존 `cell_count` 결과는 그대로 유지(기본값)

완료 조건:

- 옵션으로 정확도 향상이 제공되고, 기본 동작은 변하지 않음

### Workstream E (P1): oxidation fractional mask 가중 + BC coupling 고도화 (M4)

#### E1. oxidation에 `mask_weighting` 옵션 추가(기본값 유지)

현재 oxidation의 open/blocked 판정이 `mask_eff>0.5`로 이진화되어 edge fractional이 반영되지 않는다.

아래 옵션을 추가한다.

- `mask_weighting: "binary"` (default, 기존 동작)
- `mask_weighting: "time_scale"` (신규 권장)

Deal-Grove의 비선형성을 보존하면서 fractional을 반영하기 위해
각 x 컬럼마다 `t_eff = time_s * w(x)`로 scaling 후 `tox_new=f(tox_old, t_eff)`

`w(x)` 정의:

- `apply_on="all"`: `w=1`
- `apply_on="open"`: `w=clip(mask_eff,0,1)`
- `apply_on="blocked"`: `w=1-clip(mask_eff,0,1)`

`delta_scale(Δtox*w)`도 가능하지만,
Deal-Grove가 time에 대해 비선형이므로 `time_scale`이 더 일관적이고 단조성을 유지하기 쉽다.

추가 고려(안전장치):

- `tox_new >= tox_old` 유지
- `max(tox) > Ly` 실패 로직 유지
- `consume_dopants`/표면 이동(`delta_out`)은 `delta_tox` 기반이므로 자연스럽게 fractional 반영됨

#### E2. diffusion top BC의 "effective open" 옵션

현재 top BC는 `open_frac = mask_eff` 혼합 + `cap_eps_um`로 gate(산화막 두께 기준)한다.

여기에 옵션을 추가해 coupling을 더 자연스럽게 만들 수 있다.

- `cap_model: "hard"` (default, 기존 `tox<=cap_eps`로 open/blocked gating)
- `cap_model: "exp"` (옵션)

`open_frac_eff = open_frac * exp(-tox_um / cap_len_um)`

`cap_len_um`은 새 파라미터(예: 0.01~0.1um 범위에서 사용자가 조절)

이렇게 하면,

- oxidation으로 tox가 증가할수록 top 경계의 "유효 개방도"가 연속적으로 감소
- mask smoothing과도 정합성이 좋아짐

#### E3. 테스트

oxidation:

- 기존 테스트는 `mask_weighting` 미지정으로 그대로 통과해야 함
- 신규 테스트: smooth mask edge에서 tox profile이 더 연속적/완만해지는지

diffusion:

- all-Neumann 질량보존 테스트는 그대로 유지
- `exp cap`에서 tox 증가 -> `flux_out` 감소 경향 확인(정성 테스트)

완료 조건:

- 옵션으로 fractional/연속 coupling 제공
- 기본값은 기존과 동일

---

## 6. 리스크와 대응

### R1. 리팩터링으로 인한 미세 수치 차이

대응:

- golden metric 테스트를 tolerance 기반으로 설계
- 엔진 이전 단계(M1)에서는 "로직 동일"을 최우선, 구조 변경 최소화

### R2. GUI 요구사항(메모리/속도)과 deck 파일 I/O의 충돌

대응:

- `ArtifactSink`로 I/O를 추상화
- GUI는 `MemorySink`(또는 임시폴더)로 선택 가능하게 설계

### R3. 새 옵션 추가로 deck 스키마 복잡도 증가

대응:

- 기본값을 기존과 동일하게 두고, 문서/예제에 "옵션"으로만 소개
- GUI에는 "고급 옵션"으로 숨김 처리 가능

### R4. 의존성 추가에 대한 부담

대응:

- 새 기능은 가능한 `numpy/scipy` 범위 내에서 구현(`iso area`도 `scikit-image` 없이)
- GUI/DEV는 extras로 분리

---

## 7. 작업 체크리스트(실행용)

### M0: Reproducibility

- [x] `pyproject`에 `numpy/scipy/pyyaml/matplotlib` 의존성 명시
- [x] `[dev]` extras에 `pytest` 추가, `[gui]` extras에 `streamlit` 추가
- [x] `proc2d selfcheck` CLI 추가
- [x] README Quickstart 갱신
- [x] CI에서 `pip install -e .[dev] && pytest`

### M1: Engine 도입 + deck 실행 경로 통합

- [x] `engine.run_deck_mapping()` 추가
- [x] `deck.py`에서 "파싱/정규화"와 "실행" 분리
- [x] step executor 공통화(최소한 deck 실행이 engine을 타도록)
- [x] examples 기반 golden metric 테스트 추가

### M2: GUI 엔진화

- [x] `build_deck_from_ui()` 구현
- [x] GUI 실행 경로를 engine 호출로 교체
- [x] GUI analyze preset을 `presets.py`로 분리
- [x] schedule 고급 입력 지원(텍스트 기반 최소 구현)
- [x] GUI/Deck parity 테스트 추가

### M3: `iso_contour_area` 개선 옵션

- [x] `iso_contour_area(method="tri_linear")` 구현
- [x] analyze step에서 옵션 노출
- [x] 단위 테스트 + 수렴 테스트 추가

### M4: oxidation fractional + BC coupling 옵션

- [x] oxidation에 `mask_weighting` 추가(`binary/time_scale`)
- [x] diffusion top BC에 `cap_model` 옵션 추가(`hard/exp`)
- [x] 신규 옵션 테스트 + 기존 테스트 회귀 확인

---

## 8. 산출물(Deliverables)

### 코드

- `proc2d/engine.py` (+ `ArtifactSink`/`RunResult`)
- `deck/cli/gui` 리팩터링 반영
- `metrics/oxidation/diffusion` 옵션 확장

### 문서

- README Quickstart + 예제 업데이트
- 옵션 사용 가이드(짧은 섹션)

### 테스트/CI

- golden metric 회귀 테스트
- 신규 옵션 단위 테스트
- selfcheck + CI 안정화
