"""두 번의 501문항 실행 결과를 문항 단위로 대조해 회귀와 개선을 갈라낸다.

## 왜 필요한가

지표 총계만 비교하면 "grounded=False가 46건에서 44건으로 줄었다"까지만 알 수 있는데,
그 안에서 **10건이 고쳐지고 8건이 새로 깨진 것**과 **2건만 고쳐진 것**은 전혀 다른
상황이다. 앞은 회귀 8건을 숨긴 채 개선처럼 보이는 위험한 결과다.

그래서 총계가 아니라 **문항 번호 단위로 상태 전이**를 본다:
  - FIXED   : baseline에서 실패 → current에서 성공 (개선)
  - BROKEN  : baseline에서 성공 → current에서 실패 (회귀, 가장 중요)
  - 유지     : 양쪽 동일

## 사용법

    python eval/compare_runs.py
    python eval/compare_runs.py --baseline eval/results/baseline_20260902.jsonl \
                               --current  eval/results/eval_run.jsonl
    python eval/compare_runs.py --show-broken     # 회귀 문항 상세 출력
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASELINE = os.path.join(EVAL_DIR, "results", "baseline_20260902.jsonl")
DEFAULT_CURRENT = os.path.join(EVAL_DIR, "results", "eval_run.jsonl")


def load_run(path: str) -> dict[str, dict]:
    """no를 키로 하는 레코드 맵. 같은 no가 여러 번 있으면 마지막 것이 이긴다
    (--resume으로 이어 돌린 경우 뒤쪽이 최신)."""
    records: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            no = str(rec.get("no", ""))
            if no:
                records[no] = rec
    return records


# ── 문항 하나에서 뽑아내는 판정들 ────────────────────────────────────


def is_grounded(rec: dict) -> bool | None:
    v = rec.get("verification") or {}
    return v.get("grounded")


def unsupported_numbers(rec: dict) -> list[str]:
    v = rec.get("verification") or {}
    confirmed = v.get("unsupported_numbers_confirmed")
    if isinstance(confirmed, list):
        return confirmed
    return []


def deterministic_category(rec: dict) -> str:
    # 러너는 라우터 판정을 "router" 키에 담는다(routing 아님 — 실측으로 확인).
    return (rec.get("router") or {}).get("deterministic_category") or "해당없음"


def needs_clarification(rec: dict) -> bool:
    return bool((rec.get("flow") or {}).get("needs_clarification"))


def repair_attempted(rec: dict) -> bool:
    return bool((rec.get("flow") or {}).get("repair_attempted"))


def guardian_enabled(rec: dict) -> bool:
    g = rec.get("guardian") or {}
    return bool(g.get("enabled"))


def has_error(rec: dict) -> bool:
    return bool(rec.get("error"))


def elapsed(rec: dict) -> float:
    try:
        return float(rec.get("elapsed_sec") or 0)
    except (TypeError, ValueError):
        return 0.0


def answer_len(rec: dict) -> int:
    return len(rec.get("answer") or "")


# ── 집계 ──────────────────────────────────────────────────────────


def summarize(records: dict[str, dict], label: str) -> dict:
    total = len(records)
    grounded_false = sum(1 for r in records.values() if is_grounded(r) is False)
    grounded_none = sum(1 for r in records.values() if is_grounded(r) is None)
    with_unsupported = sum(1 for r in records.values() if unsupported_numbers(r))
    unsupported_total = sum(len(unsupported_numbers(r)) for r in records.values())
    clarify = sum(1 for r in records.values() if needs_clarification(r))
    repair = sum(1 for r in records.values() if repair_attempted(r))
    guardian = sum(1 for r in records.values() if guardian_enabled(r))
    errors = sum(1 for r in records.values() if has_error(r))
    deterministic = sum(
        1 for r in records.values() if deterministic_category(r) != "해당없음"
    )

    # 에러 문항(특히 네트워크 단절로 인한 재시도)은 응답시간 통계에서 제외한다.
    # 실측(2026-09-06): 단절 2건이 각각 5,579초/28,397초를 기록해, 나머지 498건이
    # 평균 13.3초인데도 전체 평균이 81초로 6배 왜곡됐다. "정상 응답이 얼마나
    # 걸리는가"를 보려는 지표에 재시도 소진 시간이 섞이면 안 된다.
    times = sorted(elapsed(r) for r in records.values() if elapsed(r) > 0 and not has_error(r))
    avg_time = sum(times) / len(times) if times else 0
    p95_time = times[int(len(times) * 0.95)] if times else 0

    lens = [answer_len(r) for r in records.values() if answer_len(r) > 0]
    avg_len = sum(lens) / len(lens) if lens else 0

    return {
        "label": label,
        "total": total,
        "grounded_false": grounded_false,
        "grounded_none": grounded_none,
        "with_unsupported": with_unsupported,
        "unsupported_total": unsupported_total,
        "clarify": clarify,
        "repair": repair,
        "guardian": guardian,
        "errors": errors,
        "deterministic": deterministic,
        "avg_time": avg_time,
        "p95_time": p95_time,
        "avg_len": avg_len,
    }


def print_summary_table(base: dict, curr: dict) -> None:
    rows = [
        ("문항 수", "total", "건"),
        ("에러(예외)", "errors", "건"),
        ("grounded=False", "grounded_false", "건"),
        ("grounded=None(검증없음)", "grounded_none", "건"),
        ("미지원수치 발생 문항", "with_unsupported", "건"),
        ("미지원수치 총 개수", "unsupported_total", "개"),
        ("정형 카테고리 발동", "deterministic", "건"),
        ("역질문(clarification)", "clarify", "건"),
        ("재생성(repair)", "repair", "건"),
        ("Guardian 발동", "guardian", "건"),
    ]
    print(f"\n{'지표':<28} {'baseline':>10} {'current':>10} {'변화':>10}")
    print("-" * 62)
    for name, key, unit in rows:
        b, c = base[key], curr[key]
        diff = c - b
        arrow = f"{diff:+d}" if diff else "="
        print(f"{name:<28} {b:>10} {c:>10} {arrow:>10}")

    print(f"\n{'응답시간/길이':<28} {'baseline':>10} {'current':>10} {'변화':>10}")
    print("-" * 62)
    for name, key in [("평균 응답시간(초)", "avg_time"), ("P95 응답시간(초)", "p95_time"),
                      ("평균 답변 길이(자)", "avg_len")]:
        b, c = base[key], curr[key]
        diff = c - b
        print(f"{name:<28} {b:>10.1f} {c:>10.1f} {diff:>+10.1f}")


def transitions(
    base_recs: dict[str, dict], curr_recs: dict[str, dict], predicate, name: str
) -> tuple[list[str], list[str]]:
    """predicate가 True면 '나쁨'인 상태로 본다. FIXED/BROKEN 문항 번호를 돌려준다."""
    common = set(base_recs) & set(curr_recs)
    fixed, broken = [], []
    for no in sorted(common, key=lambda x: int(x) if x.isdigit() else 0):
        b_bad = predicate(base_recs[no])
        c_bad = predicate(curr_recs[no])
        if b_bad and not c_bad:
            fixed.append(no)
        elif not b_bad and c_bad:
            broken.append(no)
    return fixed, broken


def main() -> None:
    parser = argparse.ArgumentParser(
        description="두 실행 결과를 문항 단위로 대조해 회귀와 개선을 갈라낸다"
    )
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--current", default=DEFAULT_CURRENT)
    parser.add_argument("--show-broken", action="store_true",
                        help="회귀 문항의 질문·사유를 상세 출력")
    parser.add_argument("--top", type=int, default=15,
                        help="상세 출력 시 최대 문항 수")
    args = parser.parse_args()

    if not os.path.exists(args.baseline):
        raise SystemExit(f"baseline 없음: {args.baseline}")
    if not os.path.exists(args.current):
        raise SystemExit(f"current 없음: {args.current}")

    base_recs = load_run(args.baseline)
    curr_recs = load_run(args.current)

    base = summarize(base_recs, "baseline")
    curr = summarize(curr_recs, "current")

    print("=" * 62)
    print(f"baseline: {os.path.basename(args.baseline)}  ({base['total']}문항)")
    print(f"current : {os.path.basename(args.current)}  ({curr['total']}문항)")
    print("=" * 62)

    print_summary_table(base, curr)

    # ── 문항 단위 전이: 총계가 숨기는 회귀를 드러낸다 ──
    print("\n" + "=" * 62)
    print("문항 단위 상태 전이 (총계가 가리는 회귀를 드러낸다)")
    print("=" * 62)

    checks = [
        ("grounded 실패", lambda r: is_grounded(r) is False),
        ("미지원수치 발생", lambda r: bool(unsupported_numbers(r))),
        ("역질문", needs_clarification),
        ("재생성 발동", repair_attempted),
        ("에러", has_error),
    ]

    broken_detail: dict[str, list[str]] = {}
    for name, pred in checks:
        fixed, broken = transitions(base_recs, curr_recs, pred, name)
        broken_detail[name] = broken
        print(f"\n[{name}]")
        print(f"  FIXED  (개선): {len(fixed):>3}건  {fixed[:20]}")
        print(f"  BROKEN (회귀): {len(broken):>3}건  {broken[:20]}")

    # ── 정형 카테고리 분포 변화: 신규 3종 오발동 확인 ──
    print("\n" + "=" * 62)
    print("정형 카테고리 발동 분포 (신규 카테고리 오발동 확인)")
    print("=" * 62)
    b_cat = Counter(deterministic_category(r) for r in base_recs.values())
    c_cat = Counter(deterministic_category(r) for r in curr_recs.values())
    all_cats = sorted(set(b_cat) | set(c_cat), key=lambda k: -c_cat.get(k, 0))
    print(f"{'카테고리':<32} {'baseline':>9} {'current':>9} {'변화':>8}")
    print("-" * 62)
    for cat in all_cats:
        b, c = b_cat.get(cat, 0), c_cat.get(cat, 0)
        if b == 0 and c == 0:
            continue
        diff = c - b
        mark = "  ← 신규" if b == 0 and c > 0 else ""
        print(f"{cat:<32} {b:>9} {c:>9} {diff:>+8}{mark}")

    # ── 회귀 상세 ──
    if args.show_broken:
        print("\n" + "=" * 62)
        print("회귀 문항 상세")
        print("=" * 62)
        for name, nos in broken_detail.items():
            if not nos:
                continue
            print(f"\n### {name} — {len(nos)}건")
            for no in nos[: args.top]:
                rec = curr_recs[no]
                q = (rec.get("question") or "")[:70]
                v = rec.get("verification") or {}
                issues = v.get("issues") or []
                nums = unsupported_numbers(rec)
                print(f"\n  no.{no}  {q}")
                print(f"    카테고리: {deterministic_category(rec)}")
                if nums:
                    print(f"    미지원수치: {nums[:6]}")
                if issues:
                    first = issues[0] if isinstance(issues, list) else str(issues)
                    print(f"    issues: {str(first)[:100]}")

    # ── 종합 판정 ──
    print("\n" + "=" * 62)
    total_broken = sum(len(v) for v in broken_detail.values())
    if total_broken == 0:
        print("판정: 회귀 없음")
    else:
        print(f"판정: 회귀 총 {total_broken}건 — --show-broken 으로 상세 확인 필요")
    print("=" * 62)


if __name__ == "__main__":
    main()
