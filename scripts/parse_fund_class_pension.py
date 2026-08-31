"""Cost Guard용 연금 클래스(연금저축/퇴직연금-IRP) 추출기.

## 배경

팀원이 만든 `Cost_Guard_fund_class_v2_검증보정.xlsx`를 원문(VIP한국형가치투자,
KR514X450008)과 직접 대조한 결과, 연금 클래스(C-P/C-Pe/S-P/C-P2/C-P2e/S-P2 등
"P" 접미사가 붙은 클래스)는 채널·계좌유형·비용이 전부 정확했지만, 일반 클래스
(A/C/S/C-I 등 "P" 접미사가 없는 클래스)는 채널·계좌유형이 다수 틀렸다. 게다가
틀린 값에도 `검증상태=AUTO_CORRECTED`가 붙어 있어, "검증 완료"라는 이름만으로는
신뢰할 수 없다는 게 확인됐다.

Cost Guard가 실제로 쓸 데이터는 연금 클래스뿐이므로(일반/기관/랩 클래스는 애초에
비교 대상이 아니다), 이 스크립트는 그 범위만 좁혀서 원문에서 직접, 그리고
검증 가능한 방식으로 다시 뽑는다.

## 핵심 설계: 두 출처를 코드로 조인한다

투자설명서는 클래스 정보를 최소 두 곳에 따로 담고 있고, 펀드마다 표 형식이 갈린다:

  ① 클래스 정의 목록 — "수수료미징구-오프라인-개인연금(C-P)"처럼 채널+계좌유형+
     코드가 한 문자열에 들어 있다. 문서 안에 5~6번 반복 등장한다.
  ② 보수율 숫자표("나. 집합투자기구에 부과되는 보수 및 비용") — 펀드에 따라
     - 형식 A: 클래스 전체명(①과 동일한 문자열) 뒤에 숫자 10개
     - 형식 B: 클래스 코드만 단독으로 나온 뒤 숫자 10개
     둘 중 하나다. 형식 B는 코드만으로는 계좌유형을 알 수 없다.

그래서 ①에서 코드→(채널, 계좌유형) 매핑을 먼저 만들고, ②의 숫자표는 코드로
①과 조인한다. 같은 코드가 문서 안에서 여러 번 등장해도 매핑이 일치해야 정상이라
(정의 충돌=0), 이 자체가 파싱 정합성의 1차 검증 역할을 한다.

## 실측 검증 (2026-08-31)

VIP한국형가치투자(KR514X450008) 원문을 직접 읽어 6개 연금 클래스 전부 대조:
  C-P=1.66%, C-Pe=1.26%, C-P2=1.56%, C-P2e=1.21%, S-P=1.13%, S-P2=1.12%
전부 100% 일치. (참고로 이 값은 "총보수" 표가 아니라 총보수·비용 표의 7번째
컬럼이며, 요약정보의 4개 대표 클래스 표(A/C/Ae/Ce만 있고 페이지 경계에서
클래스명이 쪼개짐)는 쓰지 않는다 — 그게 팀원 파싱이 어긋난 원인으로 추정된다.)

투자설명서 PDF 100개 전부(`scripts/extract_prospectus_text.py`로 NFD 유니코드
정규화 문제를 우회해 전량 텍스트 추출 완료 — 원인은 파일시스템의 한글 디렉터리명이
NFD 분해형으로 저장돼 있어 NFC 경로 문자열로는 못 열렸던 것)에 대해 이 파서를
실행한 결과, "정의 충돌 0건 + 오염 없음" 기준으로 **64개 펀드, 203건**이 깨끗하게
나온다(is_clean() 필터, main() 참고). 41개 파일 기준 26개 펀드/74건이었던 것에서
신규 추출된 59개 파일과, 총보수·비용 컬럼 인덱스 폴백(기타비용 칸이 빈칸이라
컬럼 수가 하나 줄어드는 펀드 대응, _pick_ter/_TER_INDEX_CANDIDATES)을 더해
늘어났다. VIP한국형가치투자 6개 클래스 값은 이 라운드에서도 100% 유지됨(회귀 없음).

나머지 43개는 파서가 커버하지 못하는 펀드로, "추가 검토 필요"(정의 충돌 있음) 3건과
"정의는 있으나 숫자 매칭 실패"로 나뉜다. 확인된 실패 유형:
  - 전치형 표: 클래스 코드가 열로 먼저 나열되고 숫자가 행 단위로 뒤따름
    (지금 로직이 가정하는 "코드 뒤 숫자" 구조와 다름). 예: KR5118201004
  - 역순 표: "숫자 블록 → 클래스명 → 다음 숫자 블록" 순서(지금까지의 다른 펀드와
    반대). 예: KR5117420097 — 총보수·비용 타당범위 검증(0.01~3%)이 이 오염을
    걸러내 신뢰 목록에서는 빠지지만, 표 자체를 파싱하지는 못한다.
  - PDF 텍스트 추출 자체가 컬럼 레이아웃을 깨뜨려 클래스명·코드·펀드코드가
    한 줄에 뒤섞인 경우. 예: KR5118420006 — 원문 텍스트 품질 문제라 정규식으로
    복구하면 안 된다.
  - 정의 반복 중 줄바꿈 위치 변형 또는 진짜 원문 오염(코드 뒤바뀜 등 1회성)이
    섞여 label 문자열 전체 비교가 충돌로 잡히는 경우. 예: KR5120420039,
    KR5194450018 — 억지로 완화하면 팀원 v2가 겪은 "이름만 검증된 오염값" 재발
    위험이 있어 정직하게 "검토 필요"로 남겨둠.

전체 실행 결과는 `scripts/_parse_fund_class_output.txt` 참고.

## 아직 안 된 것 / 다음 단계

1. 여기서 나온 결과를 fund_class 테이블 스키마(product_code, class_code,
   account_type, channel, total_expense_ratio 등)로 옮기고, Cost Guard 규칙에
   연결한다.
2. 남은 36개(검토 필요 3건 + 매칭 실패 20건 + 정의 자체 없음 13건)를 손댈지는
   Cost Guard가 요구하는 커버리지 수준에 따라 판단 — 203건이면 대다수 상품에
   대해 최소 1쌍(연금저축 vs 퇴직연금, 또는 채널 간) 비교가 가능한지부터
   확인하는 게 우선. "정의 자체 없음" 13개는 애초에 연금 클래스가 없는 일반
   펀드일 가능성이 높다(예: KR5118420006, KR5123365001) — 파서 결함이 아니라
   실제로 다룰 데이터가 없는 경우인지 원문 확인이 필요.

## 사용법

    .venv/Scripts/python.exe scripts/parse_fund_class_pension.py
"""

from __future__ import annotations

import glob
import io
import os
import re
import sys

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prospectus_check")

# 클래스 정의: "수수료(미징구|선취|후취)-채널[-유형](코드)". 표 안에서는
# "수수료선취-\n오프라인(A)"처럼 줄바꿈으로 쪼개질 수 있어 줄바꿈을 허용하되,
# 다음 "수수료"가 나오기 전까지로 폭주를 막는다.
CLASS_DEF_RE = re.compile(
    r"수수료(?:미징구|선취|후취)\s*[-–]\s*((?:(?!수수료|\().)+?)\s*\(([^\n)]+)\)",
    re.DOTALL,
)
NUM_TOKEN_RE = re.compile(r"-?\d+\.\d+|없음|실비|-")

# 섹션 번호가 "나."인 문서가 많지만 "다."인 경우도 있다(실측 KR5117420097 —
# "가. 투자자에게 직접 부과되는 수수료" 다음이 "다. 집합투자기구에 부과되는 보수 및
# 비용"으로, 그 사이의 "나."가 다른 내용을 다루는 구조). 번호 자체는 검사하지 않고
# 제목 문구만으로 앵커를 잡는다.
_FEE_SECTION_HEADER_RE = re.compile(
    r"[가-힣]\.\s*집\s*합\s*투\s*자\s*기\s*구\s*에\s*부\s*과\s*되\s*는\s*보\s*수\s*및\s*비\s*용"
)
# 다음 섹션 경계도 번호를 가리지 않는다 — 자모 하나 + "."로 시작하는 새 항목이면
# 그 지점에서 자른다(예: "라.", "마." 등 어떤 글자든).
_NEXT_SECTION_RE = re.compile(r"\n\s*[가-힣]\.\s*[가-힣]")


def classify_account_type(label: str) -> str | None:
    if "퇴직연금" in label:
        return "퇴직연금/IRP"
    if "개인연금" in label or "연금저축" in label:
        return "연금저축"
    return None


def classify_channel(label: str) -> str:
    if "온라인슈퍼" in label:
        return "온라인슈퍼"
    if "온라인" in label:
        return "온라인"
    if "직판" in label:
        return "직판"
    if "오프라인" in label:
        return "오프라인"
    return "불명"


# 진짜 클래스 코드의 형태: 영문자로 시작하고 영문자/숫자/하이픈만 쓴다. "(퇴직연금)"
# 같은 부가 표기가 코드 뒤에 괄호로 덧붙을 수 있어 그 앞부분만 검사한다.
# ⚠️ CLASS_DEF_RE는 "수수료XX-...(임의문자열)"을 관대하게 매칭해서, 펀드명("...신탁
# 1호(주식)")이나 전화번호("TEL.1588-5533") 등 "수수료" 근처에 있는 무관한 괄호까지
# 코드로 오인했다(실측: '%', '2025.03.31기준,억원', 'CJ자산운용㈜→하이자산운용㈜' 등
# 141개 후보 중 다수가 이런 오탐). 코드 형태 검증으로 이런 오탐을 원천 차단한다.
_VALID_CODE_CORE_RE = re.compile(r"^[A-Za-z][A-Za-z0-9\-]{0,9}$")


# 형태 조건(영문자 시작 + 영숫자/하이픈)은 통과하지만 실제로는 표 헤더의 영문
# 컬럼명인 경우 — 실측: "종류(Class)"의 "Class"가 코드로 오인됨.
_NON_CODE_WORDS = {"Class", "Fee", "Type", "Total"}


def _looks_like_class_code(code: str) -> bool:
    core = code.split("(")[0]
    if core in _NON_CODE_WORDS:
        return False
    return bool(_VALID_CODE_CORE_RE.match(core))


def build_code_to_definition(text: str) -> tuple[dict[str, dict], list]:
    """문서 전체에서 "전체명(코드)" 패턴을 모아 코드 -> {계좌유형, 채널, 전체명} 매핑을 만든다.

    같은 코드가 여러 번 등장해도(문서 안에 5~6번 반복) 채널·계좌유형은 항상 동일해야
    정상이다 — 이 가정 자체가 파싱 정합성 검증 역할을 한다. label 원문 전체가 아니라
    (account_type, channel) 두 핵심 필드만 비교한다 — label에 페이지 헤더 텍스트가
    섞이는 오염이 있어도(실측: "...퇴근로자퇴직급여보장법에의한...=====PAGE29/59====="),
    핵심 필드가 같으면 같은 클래스로 본다.
    """
    mapping: dict[str, dict] = {}
    conflicts = []
    for m in CLASS_DEF_RE.finditer(text):
        label_raw, code_raw = m.group(1), m.group(2)
        label = re.sub(r"\s+", "", label_raw)
        code = re.sub(r"\s+", "", code_raw)
        if not _looks_like_class_code(code):
            continue
        account_type = classify_account_type(label)
        if account_type is None:
            continue
        channel = classify_channel(label)
        entry = {"account_type": account_type, "channel": channel, "label": label}
        existing = mapping.get(code)
        if existing is not None:
            if (existing["account_type"], existing["channel"]) != (account_type, channel):
                conflicts.append((code, existing, entry))
            continue
        mapping[code] = entry
    return mapping, conflicts


def find_fee_rate_sections(text: str) -> list[str]:
    """숫자 표가 실제로 딸린 "나. 집합투자기구에 부과되는 보수 및 비용" 구간들.

    같은 문서 안에서도 공백 처리가 파트마다 달라("부과되는 보수 및 비용" vs
    "부과되는보수및비용") 글자 사이 공백을 전부 선택적으로 둔다. 이 헤더는 목차에도
    한 번 등장하므로(숫자 표 없음), 모든 등장 위치를 후보로 남긴다 — 실제 채택은
    extract_pension_classes에서 매칭 개수가 가장 많은 구간을 고른다.
    """
    sections = []
    for m in _FEE_SECTION_HEADER_RE.finditer(text):
        start = m.end()
        end_m = _NEXT_SECTION_RE.search(text[start:])
        end = start + end_m.start() if end_m else min(start + 12000, len(text))
        sections.append(text[start:end])
    return sections


# 총보수·비용(연간 %)의 타당 범위. 실측 전 펀드에서 0.01~2% 사이였다. 이 범위
# 밖이면 클래스명과 엉뚱한 숫자가 짝지어진 것으로 본다 — 실측: KR5117420097은
# 표가 "숫자 블록 → 클래스명 → 다음 클래스 숫자 블록" 순서(지금까지의 다른 펀드와
# 반대)라, 클래스명 뒤 400자에서 찾은 숫자가 실제로는 다음 클래스의 값이었고
# 결과가 0.0076%처럼 비현실적으로 작게 나왔다. 이 역순 표 형식은 아직 지원하지
# 않으므로, 범위를 벗어난 값은 조용히 버려 오염된 값을 신뢰 목록에 넣지 않는다.
_PLAUSIBLE_TER_RANGE = (0.01, 3.0)

# 총보수·비용의 컬럼 위치(7번째, 인덱스 6)가 기본값이지만, "기타비용" 칸이
# "없음" 대신 아예 빈칸으로 빠지는 펀드(예: KR5119520012)는 컬럼 수 자체가
# 하나 줄어 위치가 밀린다. 인덱스 6이 비타당하면 인접 인덱스를 순서대로
# 시도한다 — 4(총보수), 7·8(총보수비용/합성총보수비용)도 실측상 같은 값이거나
# 근접한 값이라 안전한 폴백이다.
_TER_INDEX_CANDIDATES = (6, 7, 8, 4)


def _pick_ter(nums: list[str]) -> float | None:
    for idx in _TER_INDEX_CANDIDATES:
        if idx >= len(nums):
            continue
        try:
            ter = float(nums[idx])
        except ValueError:
            continue
        if _PLAUSIBLE_TER_RANGE[0] <= ter <= _PLAUSIBLE_TER_RANGE[1]:
            return ter
    return None


def extract_numeric_rows(section: str, known_codes: set[str]) -> list[tuple[str, float]]:
    """숫자표 안에서 "코드(또는 전체명(코드))" 뒤에 오는 총보수·비용을 뽑는다."""
    results = []
    for m in CLASS_DEF_RE.finditer(section):
        code = re.sub(r"\s+", "", m.group(2))
        tail = section[m.end(): m.end() + 400]
        nums = NUM_TOKEN_RE.findall(tail)
        if len(nums) < 5:
            continue
        ter = _pick_ter(nums)
        if ter is None:
            continue
        results.append((code, ter))

    already = {c for c, _ in results}
    for code in sorted(known_codes - already, key=len, reverse=True):
        pattern = re.compile(rf"(?<![A-Za-z0-9\-]){re.escape(code)}(?![A-Za-z0-9\-])\s*\n")
        m = pattern.search(section)
        if not m:
            continue
        tail = section[m.end(): m.end() + 400]
        nums = NUM_TOKEN_RE.findall(tail)
        if len(nums) < 5:
            continue
        ter = _pick_ter(nums)
        if ter is None:
            continue
        results.append((code, ter))
    return results


def extract_pension_classes(text: str) -> tuple[list[dict], list]:
    """연금저축/퇴직연금 클래스만 코드-정의-비용을 조인해 돌려준다.

    반환값의 두 번째 요소(conflicts)가 비어있지 않으면, 이 펀드는 신뢰하지 말고
    원문을 사람이 직접 대조해야 한다 — 같은 코드에 서로 다른 정의가 잡혔다는 뜻이다.
    """
    definitions, conflicts = build_code_to_definition(text)
    if not definitions:
        return [], conflicts

    best_rows: dict[str, float] = {}
    for section in find_fee_rate_sections(text):
        rows = extract_numeric_rows(section, set(definitions.keys()))
        if len(rows) > len(best_rows):
            best_rows = dict(rows)

    results = []
    for code, ter in best_rows.items():
        if code not in definitions:
            continue
        d = definitions[code]
        results.append({
            "code": code,
            "label": d["label"],
            "account_type": d["account_type"],
            "channel": d["channel"],
            "total_expense_ratio": ter,
        })
    return results, conflicts


def is_clean(classes: list[dict], conflicts: list) -> bool:
    """Cost Guard에 바로 써도 되는 수준인지 — 충돌 없고 코드에 오염 흔적이 없어야 한다.

    코드 형태 자체는 build_code_to_definition의 _looks_like_class_code가 이미
    걸러내므로, 여기서는 label(채널·유형 서술)에 페이지 헤더 등 명백한 텍스트
    오염이 섞였는지만 추가로 본다. "C-P2"처럼 코드에 숫자가 있는 건 정상이라
    (퇴직연금형 2종 클래스) 더는 배제하지 않는다 — 예전 버전은 이 조건 때문에
    VIP한국형가치투자(원문과 100% 일치 확인됨)까지 검토 필요로 잘못 분류했다.
    """
    if not classes or conflicts:
        return False
    return not any("PAGE" in c["label"] or "=====" in c["label"] for c in classes)


def main() -> None:
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    clean_funds, clean_rows = [], 0
    dirty_funds = []

    for path in sorted(glob.glob(os.path.join(BASE, "*.txt"))):
        name = os.path.basename(path)
        if "_p1-12" in name or name.startswith("_") or "_narrow" in name:
            continue
        text = open(path, encoding="utf-8", errors="ignore").read()
        classes, conflicts = extract_pension_classes(text)
        if is_clean(classes, conflicts):
            clean_funds.append((name, classes))
            clean_rows += len(classes)
        elif classes or conflicts:
            dirty_funds.append((name, classes, conflicts))

    stdout.write(f"=== 신뢰 가능(충돌 0, 오염 없음): {len(clean_funds)}개 펀드, {clean_rows}건 ===\n")
    for name, classes in clean_funds:
        stdout.write(f"\n{name}\n")
        for c in classes:
            stdout.write(
                f"  {c['code']:8} {c['label']:35} {c['account_type']:10} "
                f"{c['channel']:8} {c['total_expense_ratio']}%\n"
            )

    stdout.write(f"\n=== 추가 검토 필요(충돌 있거나 오염 의심): {len(dirty_funds)}개 펀드 ===\n")
    for name, classes, conflicts in dirty_funds:
        stdout.write(f"  {name}: 클래스 {len(classes)}건, 정의충돌 {len(conflicts)}건\n")

    stdout.flush()


if __name__ == "__main__":
    main()
