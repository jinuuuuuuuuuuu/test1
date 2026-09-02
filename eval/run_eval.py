"""평가셋 전체를 파이프라인에 돌리고 각 단계의 판단을 통째로 기록한다.

## 왜 answer만 저장하면 안 되는가

답변이 틀렸을 때 "왜 틀렸는지"를 알려면 ①~⑤ 각 단계가 무엇을 판단했는지 봐야 한다.
평가용 API(src/api/main.py)는 요강 스키마대로 answer/think_trace만 내보내고 나머지
State는 버리는데, 그러면 사후 분석 때 재실행 말고는 방법이 없다 (501문항 재실행은
수 시간짜리 작업이라 사실상 불가능).

실제 사례: "연령별 연금소득세율 표 알려줘"가 틀렸을 때 답변만 봤다면 "라우터가
분류를 잘못했다"고 보고 라우터를 고쳤을 것이다. 중간 State를 보니 라우터는 정상이었고
candidate_categories()가 후보를 잘못 만들고 있었다 — 라우터를 고쳤다면 문제는 그대로
남았다. 그래서 판정 근거를 전부 남긴다.

## 운영 전제

- 501문항 x 파이프라인 전체(문항당 4~10회 LLM 호출) = 3,000회 이상, 5~10시간 소요
- CLOVA 429가 실측된 바 있고(llm.py 주석) 재시도 백오프가 최대 80초라 더 길어질 수 있다
- 따라서 **문항 하나 끝날 때마다 즉시 파일에 append**한다 (중간에 죽어도 손실 1문항)
- 재실행하면 이미 끝난 문항은 건너뛴다 (--resume이 기본 동작)
- 같은 출력 파일에 두 프로세스가 붙는 사고를 막기 위해 락 파일을 쓴다

## 사용법

    python eval/run_eval.py                      # 전체 실행(이어서)
    python eval/run_eval.py --limit 5            # 앞 5문항만 (스모크 테스트)
    python eval/run_eval.py --only 276,277,280   # 특정 문항만
    python eval/run_eval.py --restart            # 기존 결과 무시하고 처음부터
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.agents.graph import build_graph  # noqa: E402
from src.agents.text import normalize_user_text  # noqa: E402

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_QUESTIONS = os.path.join(EVAL_DIR, "eval_questions_500.csv")
DEFAULT_OUTPUT = os.path.join(EVAL_DIR, "results", "eval_run.jsonl")
LOCK_SUFFIX = ".lock"


# ── 결과 파일 입출력 ──────────────────────────────────────────────────


def load_done_ids(output_path: str) -> set[str]:
    """이미 처리된 question_id를 읽는다 (재개용).

    깨진 줄(쓰다 만 마지막 줄 등)은 조용히 건너뛴다 — 프로세스가 강제 종료되면
    마지막 줄이 잘릴 수 있는데, 그것 때문에 전체 재실행을 강요하면 안 된다.
    """
    if not os.path.exists(output_path):
        return set()
    done = set()
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["no"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def append_result(output_path: str, record: dict) -> None:
    """결과 한 건을 즉시 파일에 쓰고 flush한다 — 프로세스가 죽어도 여기까지는 남는다."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ── 중복 실행 방지 ────────────────────────────────────────────────────


def acquire_lock(output_path: str) -> str:
    """같은 출력 파일에 두 프로세스가 붙는 것을 막는다.

    실측 사고: 백그라운드 실행이 중복돼 두 프로세스가 같은 API 할당량을 나눠 쓰면서
    양쪽 다 rate limit에 걸린 적이 있다.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    lock_path = output_path + LOCK_SUFFIX
    if os.path.exists(lock_path):
        with open(lock_path, encoding="utf-8") as f:
            info = f.read().strip()
        raise SystemExit(
            f"이미 실행 중인 것으로 보입니다: {lock_path}\n  ({info})\n"
            f"실행 중이 아니라면 이 파일을 지우고 다시 시도하세요."
        )
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(f"pid={os.getpid()} started={datetime.now().isoformat(timespec='seconds')}")
    return lock_path


def release_lock(lock_path: str) -> None:
    try:
        os.remove(lock_path)
    except OSError:
        pass


# ── 한 문항 실행 ──────────────────────────────────────────────────────


def run_one(graph, row: dict) -> dict:
    """문항 하나를 파이프라인에 돌리고, 각 단계의 판단을 통째로 담은 레코드를 만든다.

    평가 API와 동일하게 싱글턴으로 호출한다(대화 이력 없음) — 실제 평가 조건과
    다르게 돌리면 여기서 잰 성능이 평가 결과를 예측하지 못한다.
    """
    started = time.time()
    base = {
        "no": row["no"],
        "대주제": row.get("대주제", ""),
        "세부유형": row.get("세부유형", ""),
        "난이도": row.get("난이도", ""),
        "질문유형": row.get("질문유형", ""),
        "중점평가지표": row.get("중점평가지표", ""),
        "question": row["질문"],
        "점검포인트": row.get("점검포인트", ""),
        "ran_at": datetime.now().isoformat(timespec="seconds"),
    }

    try:
        state = graph.invoke({
            "question_id": f"eval-{row['no']}",
            "question": normalize_user_text(row["질문"]),
            "conversation_history": [],
            "recommendation_profile": {},
        })
    except Exception as exc:  # noqa: BLE001 — 한 문항 실패로 전체가 멈추면 안 된다
        return {
            **base,
            "elapsed_sec": round(time.time() - started, 1),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "answer": None,
        }

    verification = state.get("verification") or {}
    return {
        **base,
        "elapsed_sec": round(time.time() - started, 1),
        "error": None,

        # ① 라우터 — 첫 단추를 잘못 끼웠는지 여기서 바로 드러난다
        "router": {
            "intent": state.get("intent"),
            "scope": state.get("scope"),
            "scope_note": state.get("scope_note"),
            "is_safe": state.get("is_safe"),
            "safety_reason": state.get("safety_reason"),
            "deterministic_category": state.get("deterministic_category"),
            "deterministic_miss_signal": state.get("deterministic_miss_signal"),
        },

        # ②③ 에이전트 — 어떤 툴을 불렀고 무엇을 근거로 삼았나
        "agents": {
            "deterministic_info": state.get("deterministic_info"),
            "product_fallback_used": state.get("product_fallback_used"),
            "tool_trace": state.get("tool_trace") or [],
            "retrieved_context": state.get("retrieved_context") or [],
            "info_draft": state.get("info_draft"),
            "product_draft": state.get("product_draft"),
        },

        # ④ 검증 — 문제를 잡고도 못 막았는지, 아예 못 잡았는지 구분하는 근거
        "verification": {
            "grounded": verification.get("grounded"),
            "issues": verification.get("issues"),
            "unsupported_numbers_confirmed": verification.get("unsupported_numbers_confirmed"),
            "l0_suspect_numbers": verification.get("l0_suspect_numbers"),
            "requirements_met": verification.get("requirements_met"),
            "missing_requirements": verification.get("missing_requirements"),
            "premise_issues": verification.get("premise_issues"),
            "clarification_mode": verification.get("clarification_mode"),
            "source_limited_mode": verification.get("source_limited_mode"),
        },

        # 경로 분기 — 재수정 루프를 탔는지, 역질문으로 빠졌는지
        "flow": {
            "response_mode": state.get("response_mode"),
            "needs_clarification": state.get("needs_clarification"),
            "repair_attempted": state.get("repair_attempted"),
            "recommendation_stage": state.get("recommendation_stage"),
            "missing_information": state.get("missing_information"),
            "clarification_questions": state.get("clarification_questions"),
        },

        # 파수꾼(Guardian) — Core 답변 뒤에 붙는 별도 경고/기회 안내가 무엇을,
        # 왜 붙였는지(또는 왜 안 붙였는지)를 남긴다. guardian_result가 없으면
        # 이 문항에서 Guardian 후보가 아예 없었다는 뜻이다.
        "guardian": state.get("guardian_result"),
        "guardian_evidence": state.get("guardian_evidence") or [],

        # ⑤ 최종 산출물
        "answer": state.get("answer"),
        "think_trace": state.get("think_trace"),
    }


# ── 진행 상황 출력 ────────────────────────────────────────────────────


def format_progress(idx: int, total: int, record: dict, elapsed_total: float) -> str:
    no = record["no"]
    took = record.get("elapsed_sec", 0)
    if record.get("error"):
        status = f"ERROR {record['error'][:45]}"
    else:
        ans = record.get("answer") or ""
        v = record.get("verification") or {}
        flags = []
        if v.get("grounded") is False:
            flags.append("grounded=F")
        if v.get("unsupported_numbers_confirmed"):
            flags.append(f"unsup={len(v['unsupported_numbers_confirmed'])}")
        if record.get("flow", {}).get("repair_attempted"):
            flags.append("repaired")
        status = f"{len(ans)}자" + (f" [{','.join(flags)}]" if flags else "")

    done_ratio = idx / total if total else 0
    eta = (elapsed_total / idx * (total - idx)) if idx else 0
    return (
        f"[{idx}/{total} {done_ratio:5.1%}] no.{no:>4} {took:5.1f}s  {status}"
        f"   (남은 예상 {eta/60:.0f}분)"
    )


# ── 메인 ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="평가셋을 파이프라인에 돌려 단계별 판단을 기록한다")
    parser.add_argument("--questions", default=DEFAULT_QUESTIONS, help="질문 CSV 경로")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="결과 JSONL 경로")
    parser.add_argument("--limit", type=int, default=None, help="앞에서부터 N문항만")
    parser.add_argument("--only", default=None, help="특정 no만 (쉼표 구분, 예: 276,280)")
    parser.add_argument("--restart", action="store_true", help="기존 결과를 무시하고 처음부터")
    args = parser.parse_args()

    with open(args.questions, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        rows = [r for r in rows if r["no"] in wanted]
    if args.limit:
        rows = rows[: args.limit]

    if args.restart and os.path.exists(args.output):
        os.rename(args.output, args.output + f".bak-{datetime.now():%Y%m%d-%H%M%S}")
        print(f"기존 결과를 백업하고 처음부터 실행합니다.")

    done = load_done_ids(args.output)
    todo = [r for r in rows if r["no"] not in done]

    print(f"질문 {len(rows)}개 / 이미 완료 {len(done & {r['no'] for r in rows})}개 / 이번에 실행 {len(todo)}개")
    print(f"결과 파일: {args.output}")
    if not todo:
        print("실행할 문항이 없습니다.")
        return

    lock_path = acquire_lock(args.output)
    started = time.time()
    try:
        graph = build_graph()
        for i, row in enumerate(todo, 1):
            record = run_one(graph, row)
            append_result(args.output, record)
            print(format_progress(i, len(todo), record, time.time() - started), flush=True)
    finally:
        release_lock(lock_path)

    total_min = (time.time() - started) / 60
    print(f"\n완료: {len(todo)}문항, {total_min:.1f}분 소요")
    print(f"다음: python eval/screen_results.py --input {args.output}")


if __name__ == "__main__":
    main()
