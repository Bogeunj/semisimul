# proc2d Python 기반 TDD 도입 계획 (`plan_TDD.md`)

작성일: 2026-03-01  
대상 저장소: `semisimul_TDD` (`proc2d`)

---

## 0. 배경

이 프로젝트는 이미 `make test` 중심의 테스트 실행 경로를 갖고 있고, Python(`pytest`, `mypy`, `ruff`) 생태계와 잘 맞아 있다.  
이번 계획의 핵심은 다음이다.

- 테스트를 "개발 비용"이 아니라 "AI 시대의 실행 가능한 명세 자산"으로 운영한다.
- 구현보다 테스트를 먼저 작성하는 Red -> Green -> Refactor 루프를 팀 표준으로 만든다.
- 기존 `make test` 동작은 깨지지 않게 유지하고, 필요한 자동화를 점진적으로 추가한다.

### 0.1 참고 텍스트(뉴스레터) 반영 원칙

| 참고 인사이트 | 이 프로젝트 적용 정책 |
|---|---|
| 테스트는 비용이 아니라 자산 | 기능 정의를 테스트로 먼저 고정하고 구현은 그 뒤에 수행 |
| 테스트를 먼저 쓰면 AI 의존도를 줄일 수 있음 | AI는 "테스트를 통과하는 구현"만 담당, 테스트 의도는 사람이 확정 |
| 테스트는 AI 도구가 바뀌어도 살아남음 | Claude/기타 모델과 무관하게 `make test` 통과를 공통 품질 게이트로 사용 |

---

## 1. 현재 기준선 (Python/pytest/make)

현재 저장소에서 바로 사용 가능한 기준은 아래와 같다.

- 기본 테스트 진입점: `make test`
  - 실질 실행: `pytest -m "not integration and not adapter and not slow"`
- 빠른 실패 루프: `make test-fast`
- 전체 회귀: `make test-all`
- 마커 체계: `unit`, `integration`, `adapter`, `slow`
- 타입 체크: `python3 -m mypy`
- 린트 도구: `ruff`(dev 의존성에 포함)
- CI: `mypy` + 코어 pytest(푸시/PR) + 전체 pytest(푸시 전용)

즉, Python 프로젝트에서 TDD를 운영하기 위한 뼈대는 이미 갖춰져 있고, **운영 규칙 + 마커 정규화 + 루프 강화**가 주 작업이다.

---

## 2. 목표와 성공 기준

### 2.1 목표

1. 신규 기능/버그 수정은 반드시 테스트 선행(TDD)으로 개발한다.
2. `make test`를 중심으로 5~10분 내 피드백 루프를 유지한다.
3. AI가 생성한 코드도 테스트를 수정하지 않고 통과시키는 방식으로 검증한다.
4. 3개월 내 회귀 버그와 PR 재작업률을 유의미하게 낮춘다.

### 2.2 Definition of Done

- 기능 PR마다 "실패 테스트(RED)" 흔적이 있다.
- PR 병합 전 `make test` 통과는 필수다.
- 경계 변경(파일 출력/CLI/GUI bridge/파이프라인)은 `integration` 또는 `adapter` 테스트를 포함한다.
- 프로덕션/주요 버그는 회귀 테스트 없이 종료하지 않는다.

---

## 3. 핵심 운영 원칙 (Python + AI 보조 코딩)

1. **테스트 우선**: 코드보다 테스트를 먼저 작성한다.
2. **행동 검증 우선**: 내부 구현이 아니라 입력/출력/상태 변화를 검증한다.
3. **작은 루프**: 한 번에 테스트 1~3개만 RED로 만들고 즉시 GREEN으로 전환한다.
4. **기존 테스트 잠금**: 새 기능 구현 시 기존 테스트를 완화/삭제/skip하지 않는다.
5. **마커 필수**: 새 테스트 파일은 `unit|integration|adapter|slow` 중 최소 1개를 명시한다.
6. **빠른 경로와 전체 경로 분리**: 개발 중엔 `make test`, 병합 전엔 `make test-all`.
7. **정적 분석 결합**: 리팩터 단계에서 `mypy`, `ruff`를 함께 실행한다.
8. **회귀 테스트 의무화**: 버그 수정은 반드시 재현 테스트부터 작성한다.
9. **테스트 품질 관리**: 해피패스-only 테스트를 지양하고 경계/실패 케이스를 포함한다.
10. **AI는 구현자, 사람은 명세자**: 테스트 의도와 최종 판단 책임은 개발자가 가진다.

---

## 4. Red -> Green -> Refactor 표준 루프

### 4.1 기본 루프 (개발 중)

```bash
# RED: 실패하는 테스트 먼저 실행
python3 -m pytest tests/<path>/test_<name>.py::test_<case> -q

# GREEN: 최소 구현 후 동일 테스트 재실행
python3 -m pytest tests/<path>/test_<name>.py::test_<case> -q

# CORE 회귀
make test
```

### 4.2 리팩터 루프 (기능 완료 직전)

```bash
make test-fast
python3 -m mypy
python3 -m ruff check proc2d tests
make test-all
```

### 4.3 make 없는 환경 fallback

```bash
python3 -m pytest -m "not integration and not adapter and not slow"
python3 -m pytest
python3 -m mypy
python3 -m ruff check proc2d tests
```

---

## 5. 테스트 레이어 전략 (pytest marker 정책)

### 5.1 레이어 정의

- `unit`: 빠르고 고립된 함수/모듈 테스트 (기본 코어 루프 핵심)
- `integration`: 파이프라인/다중 모듈 조합 검증
- `adapter`: CLI/GUI bridge/외부 경계 계약 검증
- `slow`: 고비용 테스트, 기본 루프 제외

### 5.2 실행 정책

- 로컬 기본: `make test`
- 빠른 실패: `make test-fast`
- 전체 회귀: `make test-all`
- 선택 실행 예시:

```bash
python3 -m pytest -m "unit"
python3 -m pytest -m "integration or adapter"
python3 -m pytest -m "slow"
```

### 5.3 마커 정규화 (1~2주 내)

현재 일부 테스트 파일은 명시 마커가 없다. 아래 순서로 정규화한다.

1. `unit` 후보 파일부터 `pytestmark` 추가 (리스크 최소)
2. 통합 성격 테스트는 `integration` 명시
3. CLI/GUI bridge 성격 테스트는 `adapter` 명시
4. 실행시간이 긴 테스트는 `slow` 명시 후 분리 실행

결과 목표: `make test` 실행 세트의 의도가 파일 수준에서도 일관되게 보이게 만든다.

---

## 6. Makefile 기반 점진 도입 로드맵

`make test` 의미를 깨지 않는 것을 최우선으로 한다.

### Phase 0 (즉시) - 규칙 도입

- 현행 명령을 팀 표준으로 고정: `make test`, `make test-fast`, `make test-all`
- PR 템플릿에 TDD 체크박스 추가
- "테스트 수정으로 통과시키기" 금지 규칙 도입

### Phase 1 (1~2주) - Python TDD 루프 강화

- 문서에 단일 테스트 노드 실행 규칙 명시
- `mypy`/`ruff`를 Refactor 단계 필수 체크로 지정
- 회귀 버그 대응 프로세스 추가 (재현 테스트 -> 수정 -> 전체 회귀)

### Phase 2 (1개월) - 자동화 확장 (선택)

아래는 "제안"이며, 실제 적용은 별도 PR로 수행한다.

- `make test-one TEST=tests/...::test_name`
- `make test-k K=<expr>`
- `make typecheck` (`python3 -m mypy`)
- `make lint` (`python3 -m ruff check proc2d tests`)
- `make test-cov` (`python3 -m pytest tests --cov=proc2d --cov-report=term-missing --cov-report=xml`)

### Phase 3 (3개월) - 품질 게이트 정착

- CI에서 린트/타입체크/코어 테스트 실패 시 머지 차단
- slow 테스트는 별도 스케줄 또는 릴리즈 전 게이트로 운영
- 플래키 테스트 분리/수정 SLA 운영

---

## 7. AI 코딩 가드레일 (중요)

### 7.1 금지 항목

- 실패 테스트를 `skip/xfail`로 우회
- assertion 완화/삭제로 통과 유도
- 실패 테스트를 제외하도록 마커/명령 조작
- `except Exception: pass` 형태의 무차별 예외 삼키기

### 7.2 필수 항목

- RED 증거: 어떤 테스트가 먼저 실패했는지 PR에 기록
- GREEN 증거: 어떤 최소 변경으로 통과했는지 기록
- 범위 증거: `make test`와 필요한 추가 테스트 실행 결과 기록

### 7.3 README.md/팀 규칙에 넣을 3줄

1. 테스트 파일은 원칙적으로 수정하지 않는다(명세 변경 PR 제외).
2. 새 기능 구현 시 기존 테스트를 깨뜨리지 않는다.
3. 테스트 실패는 구현을 수정해 해결한다(테스트 완화 금지).

---

## 8. 파일럿 적용 우선순위 (이 저장소 맞춤)

### 8.1 1차 파일럿 (2주)

1. `proc2d/config/*` 파서/검증 테스트 강화
2. `proc2d/pipeline/*` 에러 컨텍스트/스케줄 검증 강화
3. `proc2d/export/*` 기본 산출물 계약 테스트 강화

### 8.2 2차 파일럿 (1개월)

1. CLI 실패 경로 테스트 (`tests/adapters/test_cli_adapter.py` 확장)
2. GUI bridge deck 생성 검증 (`proc2d/gui_bridge.py` 중심)
3. oxidation/cap model 회귀 보강

### 8.3 우선 추가 권장 테스트 예시

- parser가 잘못된 step payload를 즉시 거부하는지
- pipeline unknown step/type 에러 메시지가 충분한지
- export CSV 기본 linecut 자동 생성이 유지되는지
- CLI에서 DeckError 발생 시 종료코드/메시지가 일관적인지
- GUI bridge schedule 파싱 실패가 명확히 노출되는지

---

## 9. KPI 및 목표치

### 9.1 핵심 KPI

- 품질: 코어 테스트 통과율, 회귀 버그 재발률, 플래키 비율
- 속도: `make test` 실행시간(p50/p95), CI 피드백 시간, Red->Green MTTR
- 도입: 테스트 동반 PR 비율, TDD 체크리스트 준수율, AI 가드레일 위반 건수

### 9.2 단계별 목표

| 기간 | 품질 목표 | 속도 목표 | 도입 목표 |
|---|---|---|---|
| 2주 | 코어 통과율 >= 95% | `make test` p95 10분 이내 | 테스트 동반 PR >= 60% |
| 1개월 | 회귀 버그율 20% 감소 | CI 피드백 15% 단축 | 테스트 동반 PR >= 75% |
| 3개월 | 회귀 버그율 40% 감소 | Red->Green MTTR 4시간 이내 | 테스트 동반 PR >= 90% |

---

## 10. 오늘 바로 실행 (15분 / 1분 / 10분)

### 10.1 15분 - 다음 기능을 테스트부터 작성

- 신규 기능 1개를 선택하고 실패 테스트 2~3개를 먼저 작성
- 실행:

```bash
python3 -m pytest tests/<target_file>.py::test_<new_behavior> -q
```

### 10.2 1분 - AI 테스트 규칙 고정

- 팀 문서(예: `README.md`)에 "테스트 완화 금지" 3줄 추가

### 10.3 10분 - 회귀 테스트 1개 추가

- 현재 가장 중요한 기존 기능 1개를 선택
- 실패 재현 테스트 또는 계약 테스트 1개를 추가
- 실행:

```bash
make test-fast
make test
```

---

## 11. PR 체크리스트 (운영용)

- [ ] RED(실패 테스트)부터 시작했다.
- [ ] GREEN은 최소 구현으로 통과시켰다.
- [ ] Refactor 후 `make test`를 다시 통과했다.
- [ ] 필요 시 `make test-all`까지 확인했다.
- [ ] `python3 -m mypy` 실행 결과를 확인했다.
- [ ] `python3 -m ruff check proc2d tests` 실행 결과를 확인했다.
- [ ] 테스트를 약화해 통과시키는 변경이 없다.
- [ ] 버그 수정이면 회귀 테스트를 포함했다.

---

## 12. 결론

Python 프로젝트에서 AI를 잘 활용하는 핵심은 "구현을 빨리 짜는 것"이 아니라,  
`make test`를 중심으로 테스트 명세를 먼저 고정하고 그 명세를 지키게 만드는 운영 체계다.

이 계획은 현재 저장소의 실행 방식(`make test` + `pytest marker` + `mypy`)을 유지한 채,  
리스크 낮게 TDD를 정착시키기 위한 실무형 로드맵이다.
