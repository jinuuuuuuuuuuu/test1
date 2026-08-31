"""투자설명서 PDF 100개 전체를 텍스트로 추출한다.

## 배경 — 이전에 막혔던 지점과 실제 원인

이전 세션들에서 `data/raw/prospectus/투자설명서/<코드>/*.pdf`를 Read 도구나
직접 만든 pypdf 경로 문자열로 열려고 하면 "파일이 존재하지 않음" 오류가 반복
됐다. PowerShell Get-ChildItem으로는 파일이 보이는데 Python에서 못 여는
비대칭이 있어 "인코딩 문제"로만 짐작하고 매번 우회해왔다.

이번에 원인을 직접 진단했다: 이 폴더 트리의 한글 디렉터리명이 **NFD(유니코드
분해형)**로 저장돼 있다 — 예를 들어 "투"라는 한 글자가 자모 3개로 분해되어
`unicodedata.normalize('NFC', name) != name`이 성립한다. Read 도구나 사람이
타이핑한 경로 문자열은 보통 NFC(완성형)라, 파일시스템의 실제 바이트열과
안 맞아서 "파일 없음"이 났던 것이다.

**해결책은 간단하다**: 경로 문자열을 직접 타이핑/하드코딩하지 않고,
`os.listdir()`로 얻은 이름을 그대로(정규화하지 않고) 이어 붙여서 연다.
`os.listdir()`은 파일시스템이 실제로 갖고 있는 바이트열을 그대로 돌려주므로
언제나 존재하는 경로가 나온다. 이 스크립트가 그 방식을 쓴다.

⚠️ 콘솔에 한글을 print하면 여전히 cp949 인코딩 에러가 날 수 있다(파일 자체와는
무관한 별개 문제) — 이 스크립트는 콘솔 출력을 최소화하고 파일로 직접 쓴다.

## 사용법

    .venv/Scripts/python.exe scripts/extract_prospectus_text.py

이미 prospectus_check/*.txt로 텍스트가 있는 41개는 건너뛰고, 나머지 PDF만
prospectus_check/<코드>.txt로 새로 추출한다. --all로 전체 재추출.
"""

from __future__ import annotations

import argparse
import os
import sys

import pypdf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_ROOT = os.path.join(REPO_ROOT, "data", "raw", "prospectus")
OUT_DIR = os.path.join(REPO_ROOT, "prospectus_check")


def find_pdf_files() -> list[tuple[str, str]]:
    """(코드, PDF 절대경로) 목록을 100개 전부 돌려준다.

    os.listdir()이 반환한 이름을 그대로(정규화 없이) 경로 조합에 쓴다 — 여기서
    NFC로 정규화하면 다시 파일을 못 여는 문제가 재현된다.
    """
    results = []
    # PDF_ROOT 바로 아래 "투자설명서" 폴더 하나가 있다 — 이름 자체가 한글이라
    # listdir로 실제 바이트열을 얻어야 한다.
    entries = os.listdir(PDF_ROOT)
    if not entries:
        return results
    inner = os.path.join(PDF_ROOT, entries[0])
    for code_dir_name in sorted(os.listdir(inner)):
        code_dir = os.path.join(inner, code_dir_name)
        if not os.path.isdir(code_dir):
            continue
        pdf_names = [f for f in os.listdir(code_dir) if f.lower().endswith(".pdf")]
        if not pdf_names:
            continue
        # code_dir_name 자체가 코드지만, 파일명에서 한 번 더 확인한다.
        results.append((code_dir_name, os.path.join(code_dir, pdf_names[0])))
    return results


def extract_text(pdf_path: str) -> str:
    reader = pypdf.PdfReader(pdf_path)
    parts = []
    for i, page in enumerate(reader.pages, start=1):
        parts.append(f"\n===== PAGE {i}/{len(reader.pages)} =====\n")
        parts.append(page.extract_text() or "")
    return "".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="이미 있는 텍스트도 다시 추출")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    pdfs = find_pdf_files()

    log_path = os.path.join(REPO_ROOT, "scripts", "_extract_prospectus_log.txt")
    done, skipped, failed = [], [], []

    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"PDF 총 {len(pdfs)}개 발견\n\n")
        for code, pdf_path in pdfs:
            out_path = os.path.join(OUT_DIR, f"{code}.txt")
            if os.path.exists(out_path) and not args.all:
                skipped.append(code)
                continue
            try:
                text = extract_text(pdf_path)
            except Exception as exc:  # noqa: BLE001 - 배치 추출이므로 하나 실패해도 계속 진행
                failed.append((code, f"{type(exc).__name__}: {exc}"))
                log.write(f"[실패] {code}: {type(exc).__name__}: {exc}\n")
                continue
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            done.append(code)
            log.write(f"[완료] {code}: {len(text)}자\n")

        log.write(f"\n=== 요약 ===\n")
        log.write(f"신규 추출: {len(done)}개\n")
        log.write(f"기존 스킵: {len(skipped)}개\n")
        log.write(f"실패: {len(failed)}개\n")
        for code, err in failed:
            log.write(f"  실패상세: {code}: {err}\n")

    sys.stdout.write(f"완료. 로그: {log_path}\n")
    sys.stdout.write(f"신규 {len(done)} / 스킵 {len(skipped)} / 실패 {len(failed)}\n")


if __name__ == "__main__":
    main()
