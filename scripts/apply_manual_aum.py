"""수기 검수한 AUM 값을 aum_report.csv에서 읽어 prospectus.db에 반영한다.

scripts/extract_aum.py가 자동 추출에 실패한 행(status=PARSE_FAIL/NOT_FOUND)을 사람이 원문
대조해 채운 뒤 이 스크립트를 돌린다. 검수자는 CSV에서 두 칸만 채우면 된다:

  - aum_krw_million : 최신 회계기수 자본총계(=순자산총액), **백만원 단위**
  - aum_base_date   : 그 기수의 결산일 (YYYY-MM-DD)

원문에 데이터가 없으면(신규 설정 펀드의 "해당사항없음" 등) 두 칸을 비워 두면 된다 —
값이 비어 있는 행은 건너뛴다. status 열은 참고용이며 판정에 쓰지 않는다.

단위 실수(원 단위를 그대로 입력)는 자동 감지해 백만원으로 환산하고 로그로 알린다.
--dry-run으로 먼저 확인한 뒤 실제 반영하는 것을 권장한다.

사용법: python scripts/apply_manual_aum.py --dry-run
        python scripts/apply_manual_aum.py
"""

import argparse
import csv
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.storage.schema import connect

DB_PATH = "data/processed/prospectus.db"
REPORT_PATH = "data/processed/aum_report.csv"

# 원 단위로 잘못 입력한 값 감지 임계 — 백만원 단위로 1,000만(=10조원)을 넘는 펀드는 없다.
_WON_UNIT_THRESHOLD = 10_000_000
# 적재된 70건의 실측 범위(627백만원 ~ 4.78조원)를 감안한 상식 범위 검사.
_MIN_PLAUSIBLE = 10.0
_MAX_PLAUSIBLE = 10_000_000.0


def _parse_value(raw: str) -> float | None:
    text = (raw or "").replace(",", "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="DB에 쓰지 않고 반영 예정 내용만 출력")
    parser.add_argument("--csv", default=REPORT_PATH, help=f"검수본 CSV 경로 (기본: {REPORT_PATH})")
    args = parser.parse_args()

    with open(args.csv, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    applied, skipped, errors = [], [], []
    for row in rows:
        code = row["product_code"]
        status = (row.get("status") or "").strip().upper()
        value = _parse_value(row.get("aum_krw_million", ""))
        base_date = (row.get("aum_base_date") or "").strip()

        # 값이 채워져 있으면 적용 대상으로 본다 — status 문자열에 의존하지 않는다.
        # (검수자가 MANUAL_OK/OK 중 무엇을 쓰든, 오타가 있어도 값 유무로 판단하는 편이
        #  안전하다. 실측: 검수본이 28건을 "OK", 2건을 "MAUAL_NONE"으로 표기했다.)
        if value is None:
            reason = "데이터 없음으로 표기됨" if "NONE" in status else f"값 비어 있음 (status={status or '없음'})"
            skipped.append((code, reason))
            continue
        if len(base_date) != 10 or base_date[4] != "-" or base_date[7] != "-":
            errors.append((code, f"aum_base_date 형식 오류(YYYY-MM-DD 필요): {base_date!r}"))
            continue

        note = ""
        if value > _WON_UNIT_THRESHOLD:
            value = value / 1_000_000
            note = " (원 단위로 입력된 것으로 보고 백만원으로 환산)"
        if not (_MIN_PLAUSIBLE <= value <= _MAX_PLAUSIBLE):
            errors.append((code, f"상식 범위를 벗어난 값: {value:,.1f}백만원"))
            continue

        applied.append((code, value, base_date, (row.get("period") or "").strip(), note))

    for code, value, base_date, period, note in applied:
        print(f"  적용{'(예정)' if args.dry_run else ''}: {code} {value:>14,.1f}백만원 {base_date} {period}{note}")
    for code, reason in skipped:
        print(f"  건너뜀: {code} — {reason}")
    for code, reason in errors:
        print(f"  오류: {code} — {reason}")

    if not args.dry_run and applied:
        conn = connect(DB_PATH)
        with conn:
            for code, value, base_date, period, _ in applied:
                cursor = conn.execute(
                    "UPDATE fund_master SET aum_krw_million = ?, aum_base_date = ?, aum_period_label = ? "
                    "WHERE product_code = ?",
                    (value, base_date, period or None, code),
                )
                if cursor.rowcount == 0:
                    print(f"  경고: {code} — DB에 없는 상품코드")
        total = conn.execute(
            "SELECT COUNT(*) FROM fund_master WHERE aum_krw_million IS NOT NULL"
        ).fetchone()[0]
        conn.close()
        print(f"\nDB 반영 완료 — 현재 AUM 보유: {total}/100")
    else:
        print(f"\n{'[dry-run] ' if args.dry_run else ''}적용 {len(applied)}건 / 건너뜀 {len(skipped)}건 / 오류 {len(errors)}건")


if __name__ == "__main__":
    main()
