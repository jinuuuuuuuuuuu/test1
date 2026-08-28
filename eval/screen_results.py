"""run_eval.py 결과에서 "사람이 직접 봐야 할 문항"을 골라낸다.

## 왜 필요한가

501문항 답변은 15만~40만 자라 전부 읽을 수 없다. 그렇다고 무작위로 뽑아 읽으면
정작 틀린 답변을 놓친다. 그래서 **정답을 몰라도 판정 가능한 위험 신호**로 먼저
후보를 좁히고, 사람은 그것만 집중해서 읽는다.

## 이 스크립트가 하는 일 (채점이 아니라 분류다)

여기서 "FLAG"가 붙었다고 틀린 답변이라는 뜻이 아니다 — "확인이 필요하다"는 뜻이다.
반대로 플래그가 없다고 맞는 답변이라는 보장도 없어서, 통과분에서도 무작위 표본을
뽑아 함께 검수 대상에 넣는다(스크리닝이 놓치는 유형을 잡기 위해).

## 위험 신호 종류

  error           — 파이프라인이 아예 실패
  no_answer       — 답변이 비었거나 너무 짧음
  unsupported_leak— ④가 "근거에 없다"고 확정한 수치가 최종 답변에 남아 있음 (가장 심각)
  grounded_false  — ④가 근거 부실로 판정
  missing_no_disc — 답 못한 항목이 있는데 한계 고지 문구가 없음
  no_evidence     — 근거 0건인데 답변이 긺 (지어냈을 가능성)
  expected_number — 점검포인트에 적힌 기대 수치가 답변에 없음
  scope_suspect   — 연금 관련 질문인데 범위외로 판정
  unsafe_suspect  — 안전성 차단 기대 문항인데 통과됨
  repaired        — ④ 탈락 후 재실행됨 (품질 의심 신호, 단독으로는 약함)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(EVAL_DIR, "results", "eval_run.jsonl")
DEFAULT_QUESTIONS = os.path.join(EVAL_DIR, "eval_questions_500.csv")

# 한계 고지 표현 — verification.py의 _LIMIT_DISCLOSURE_MARKERS와 같은 사상.
# 여기서 목록을 따로 두는 이유: 스크리닝은 파이프라인 코드에 의존하지 않고 결과만
# 보고 판정해야, 파이프라인이 바뀌어도 과거 결과를 같은 기준으로 다시 잴 수 있다.
_LIMIT_MARKERS = (
    "확인이 어렵", "확인하기 어렵", "확인되지 않", "확인할 수 없",
    "제공된 자료", "보유한 자료", "자료에 없", "자료에는 없",
    "포함되어 있지 않", "안내드리기 어렵", "답변드리기 어렵", "알 수 없",
)

# 점검포인트에서 "이 수치가 답에 나와야 한다"를 뽑는다. 단위 없는 맨 숫자는 잡지
# 않는다 — "2022년 시행"의 연도나 "600/900만원"의 앞 숫자처럼 기대값이 아닌 것이
# 섞이면 오탐이 늘어 스크리닝 신뢰도가 떨어진다.
_EXPECTED_NUM_RE = re.compile(r"\d[\d,]*(?:\.\d+)?\s?(?:%|만원|억원|등급)")

# 점검포인트가 "차단/범위외를 기대한다"고 적어둔 경우를 읽는다(기대값 컬럼 대용).
_EXPECT_BLOCK_MARKERS = ("is_safe=false", "차단", "거부", "불가 안내", "조회불가", "불법")
_EXPECT_OUT_OF_SCOPE_MARKERS = ("범위외", "범위 밖", "범위밖")


def _norm_num(token: str) -> str:
    return token.replace(",", "").replace(" ", "")


# 수치 직후에 오는 부정·교정 표현. 이게 붙으면 그 수치를 사실로 주장하는 게 아니라
# "틀렸다"고 바로잡는 문맥이므로 위반으로 세지 않는다.
_NEGATION_MARKERS = (
    "가 아니", "이 아니", "은 아니", "는 아니", "아닙니다", "아니라",
    "가 아닌", "이 아닌", "잘못", "오해", "사실과 다",
)


def _is_asserted(answer: str, number: str) -> bool:
    """답변이 그 수치를 '사실로 주장'하는지 판정한다 (부정 문맥이면 False).

    수치가 등장한 각 위치에서 뒤쪽 25자를 보고 부정 표현이 있는지 확인한다.
    한 곳이라도 부정 없이 쓰였으면 주장으로 본다.
    """
    norm_answer = _norm_num(answer)
    norm_num = _norm_num(number)
    start = 0
    while True:
        idx = norm_answer.find(norm_num, start)
        if idx == -1:
            return False
        tail = norm_answer[idx + len(norm_num): idx + len(norm_num) + 25]
        if not any(m.replace(" ", "") in tail for m in _NEGATION_MARKERS):
            return True  # 부정 없이 그대로 쓰인 곳이 있다
        start = idx + len(norm_num)


def load_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def screen_one(rec: dict) -> list[str]:
    """레코드 하나에서 위험 신호 목록을 만든다 (정답을 몰라도 판정 가능한 것만)."""
    flags: list[str] = []
    answer = rec.get("answer") or ""
    v = rec.get("verification") or {}
    agents = rec.get("agents") or {}
    router = rec.get("router") or {}
    flow = rec.get("flow") or {}
    checkpoint = (rec.get("점검포인트") or "").lower()

    if rec.get("error"):
        return ["error"]
    if len(answer.strip()) < 30:
        flags.append("no_answer")

    # ④가 "근거에 없다"고 확정한 수치가 최종 답변에 그대로 남았는지 — 가장 심각한 신호.
    # (⑤ 프롬프트에 이 목록을 넘기도록 고쳤으므로, 여기 걸리면 그 조치가 안 통한 것이다)
    #
    # ⚠️ 단순 문자열 포함으로 판정하면 오탐이 난다. 실측(no.5): 답변이 "평균 임금의
    # 60%가 아니라 30일분에 계속근로기간을 곱하여"처럼 **틀린 수치를 부정하는 문맥**으로
    # 인용했는데 위반으로 잡혔다 — 이건 오히려 올바른 교정 답변이다. 수치 뒤에 부정
    # 표현이 이어지면 사실 주장으로 보지 않는다.
    confirmed = v.get("unsupported_numbers_confirmed") or []
    leaked = [n for n in confirmed if _is_asserted(answer, n)]
    if leaked:
        flags.append("unsupported_leak")

    if v.get("grounded") is False:
        flags.append("grounded_false")

    missing = v.get("missing_requirements") or []
    if missing and not any(m in answer for m in _LIMIT_MARKERS):
        flags.append("missing_no_disc")

    if not (agents.get("retrieved_context") or []) and len(answer) > 200:
        flags.append("no_evidence")

    # 점검포인트에 명시된 기대 수치가 답변에 없으면 확인 대상
    expected = {_norm_num(t) for t in _EXPECTED_NUM_RE.findall(rec.get("점검포인트") or "")}
    if expected and not any(e in _norm_num(answer) for e in expected):
        flags.append("expected_number")

    # 점검포인트가 차단/범위외를 기대한다고 적었는데 통과된 경우
    if any(m in checkpoint for m in _EXPECT_BLOCK_MARKERS) and router.get("is_safe") is not False:
        flags.append("unsafe_suspect")
    if (
        any(m in checkpoint for m in _EXPECT_OUT_OF_SCOPE_MARKERS)
        and router.get("scope") == "범위내"
    ):
        flags.append("scope_suspect")
    # 반대 방향: 범위외 판정인데 점검포인트가 범위외를 기대하지 않은 경우
    if router.get("scope") == "범위외" and not any(
        m in checkpoint for m in _EXPECT_OUT_OF_SCOPE_MARKERS
    ):
        flags.append("scope_suspect")

    if flow.get("repair_attempted"):
        flags.append("repaired")

    return flags


def summarize(records: list[dict]) -> None:
    total = len(records)
    errors = sum(1 for r in records if r.get("error"))
    times = [r.get("elapsed_sec") or 0 for r in records if not r.get("error")]
    repaired = sum(1 for r in records if (r.get("flow") or {}).get("repair_attempted"))
    grounded_false = sum(
        1 for r in records if (r.get("verification") or {}).get("grounded") is False
    )
    det = sum(1 for r in records if (r.get("agents") or {}).get("deterministic_info"))
    no_ev = sum(1 for r in records if not ((r.get("agents") or {}).get("retrieved_context") or []))

    print(f"\n{'=' * 68}\n전체 통계 ({total}문항)\n{'=' * 68}")
    print(f"  실패(error)      : {errors} ({errors/total:.1%})")
    if times:
        print(f"  응답시간         : 평균 {sum(times)/len(times):.1f}s / 최대 {max(times):.1f}s / 총 {sum(times)/60:.0f}분")
    print(f"  재수정(repair)   : {repaired} ({repaired/total:.1%})")
    print(f"  grounded=False   : {grounded_false} ({grounded_false/total:.1%})")
    print(f"  정형답변 경로    : {det} ({det/total:.1%})")
    print(f"  근거 0건         : {no_ev} ({no_ev/total:.1%})")

    print(f"\n  [대주제별 문항수]")
    for k, n in Counter(r.get("대주제", "?") for r in records).most_common():
        print(f"    {k}: {n}")

    print(f"\n  [라우터 카테고리 확정 분포]")
    cats = Counter((r.get("router") or {}).get("deterministic_category") or "?" for r in records)
    for k, n in cats.most_common(10):
        print(f"    {k}: {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="평가 결과에서 검수 대상을 골라낸다")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--sample", type=int, default=50, help="통과분에서 추가로 뽑을 무작위 표본 수")
    parser.add_argument("--seed", type=int, default=42, help="표본 추출 시드 (재현성)")
    parser.add_argument("--out", default=None, help="검수 대상을 저장할 JSONL (기본: <input>.review.jsonl)")
    args = parser.parse_args()

    records = load_records(args.input)
    if not records:
        raise SystemExit(f"결과가 비어 있습니다: {args.input}")

    summarize(records)

    flagged, clean = [], []
    for rec in records:
        flags = screen_one(rec)
        # repaired 단독은 약한 신호라 검수 대상으로 올리지 않는다 (통계로만 본다)
        strong = [f for f in flags if f != "repaired"]
        rec["_flags"] = flags
        (flagged if strong else clean).append(rec)

    print(f"\n{'=' * 68}\n위험 신호별 건수\n{'=' * 68}")
    counts = Counter(f for r in records for f in r["_flags"])
    for k, n in counts.most_common():
        mark = "  (약한 신호 — 통계만)" if k == "repaired" else ""
        print(f"  {k:18} {n:4}{mark}")

    random.seed(args.seed)
    sampled = random.sample(clean, min(args.sample, len(clean)))
    review = flagged + sampled

    print(f"\n{'=' * 68}\n검수 대상\n{'=' * 68}")
    print(f"  위험 신호 걸림 : {len(flagged)}")
    print(f"  통과분 무작위  : {len(sampled)} (스크리닝이 놓치는 유형 확인용)")
    print(f"  합계           : {len(review)} / {len(records)}")

    out_path = args.out or (args.input + ".review.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in review:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n저장: {out_path}")

    print(f"\n{'=' * 68}\n가장 심각한 신호 (unsupported_leak / error / no_answer)\n{'=' * 68}")
    critical = [r for r in flagged if {"unsupported_leak", "error", "no_answer"} & set(r["_flags"])]
    if not critical:
        print("  없음")
    for r in critical[:20]:
        print(f"  no.{r['no']:>4} [{','.join(r['_flags'])}] {r['question'][:50]}")


if __name__ == "__main__":
    main()
