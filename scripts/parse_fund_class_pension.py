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

45개 텍스트 파일(prospectus_check/*.txt) 중 "정의 충돌 0건 + 코드에 이상 문자
없음" 기준으로 걸러 **14개 펀드, 43개 연금 클래스**가 깨끗하게 나온다(CLEAN_RESULTS
필터, main() 참고). 나머지 31개는 표 형식 변형(줄바꿈 위치, 페이지 헤더 텍스트가
클래스명에 섞이는 오염 등)이 더 다양해서 이 정규식이 아직 못 따라간다 — 무리하게
정규식을 넓히면 팀원의 v2가 겪은 것과 같은 "이름만 검증된 오염값"이 재발하므로,
지금은 안전한 14개만 신뢰하고 나머지는 사람이 원문 대조하는 편이 낫다.

## 아직 안 된 것 / 다음 단계

1. 14개 → 31개로 수율 올리기: 실패 유형별로 정규식 분기 추가
   (`no_numeric_match` 출력의 코드 목록에서 페이지헤더/숫자 오염 패턴 확인)
2. prospectus_check/*.txt는 45개 펀드만 커버한다. 나머지 55개는 PDF에서 텍스트를
   새로 뽑아야 하는데, 이 프로젝트 환경에서 한글 경로의 PDF를 Read/pypdf로 직접
   여는 게 실패한 전례가 있다(파일시스템 인코딩 문제로 추정) — PowerShell
   Get-ChildItem으로 존재 확인은 되지만 파일 열기 자체가 막힌다. ASCII 경로로
   복사한 뒤 처리하는 우회가 필요하다.
3. 여기서 나온 결과를 fund_class 테이블 스키마(product_code, class_code,
   account_type, channel, total_expense_ratio 등)로 옮기고, Cost Guard 규칙에
   연결한다.

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

_FEE_SECTION_HEADER_RE = re.compile(
    r"나\.\s*집\s*합\s*투\s*자\s*기\s*구\s*에\s*부\s*과\s*되\s*는\s*보\s*수\s*및\s*비\s*용"
)
_NEXT_SECTION_RE = re.compile(r"\n\s*다\.\s*[가-힣]")


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


def build_code_to_definition(text: str) -> tuple[dict[str, dict], list]:
    """문서 전체에서 "전체명(코드)" 패턴을 모아 코드 -> {계좌유형, 채널, 전체명} 매핑을 만든다.

    같은 코드가 여러 번 등장해도(문서 전체에 5~6번 반복) 계좌유형/채널은 항상
    동일해야 정상이다 — 이 가정 자체가 파싱 정합성 검증 역할을 한다.
    """
    mapping: dict[str, dict] = {}
    conflicts = []
    for m in CLASS_DEF_RE.finditer(text):
        label_raw, code_raw = m.group(1), m.group(2)
        label = re.sub(r"\s+", "", label_raw)
        code = re.sub(r"\s+", "", code_raw)
        account_type = classify_account_type(label)
        if account_type is None:
            continue
        channel = classify_channel(label)
        entry = {"account_type": account_type, "channel": channel, "label": label}
        if code in mapping and mapping[code] != entry:
            conflicts.append((code, mapping[code], entry))
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


def extract_numeric_rows(section: str, known_codes: set[str]) -> list[tuple[str, float]]:
    """숫자표 안에서 "코드(또는 전체명(코드))" 뒤에 오는 총보수·비용(7번째 숫자)을 뽑는다."""
    results = []
    for m in CLASS_DEF_RE.finditer(section):
        code = re.sub(r"\s+", "", m.group(2))
        tail = section[m.end(): m.end() + 400]
        nums = NUM_TOKEN_RE.findall(tail)
        if len(nums) < 7:
            continue
        try:
            ter = float(nums[6])
        except ValueError:
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
        if len(nums) < 7:
            continue
        try:
            ter = float(nums[6])
        except ValueError:
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
    """Cost Guard에 바로 써도 되는 수준인지 — 충돌 없고 코드에 오염 흔적이 없어야 한다."""
    if not classes or conflicts:
        return False
    return not any(
        len(c["code"]) > 12 or "PAGE" in c["label"] or any(ch.isdigit() for ch in c["code"])
        for c in classes
    )


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
