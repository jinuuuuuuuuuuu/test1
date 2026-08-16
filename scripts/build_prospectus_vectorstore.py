"""투자설명서 서술형 벡터DB(chroma_prospectus/prospectus_text) 적재 스크립트.

투자목적/투자전략(검수 DB) + 투자위험(PDF 요약 위험표)을 임베딩한다. 실제
CLOVASTUDIO_API_KEY 필요. rate limit 지연 때문에 수십 분 걸릴 수 있으며, 중간에 끊겨도
재실행하면 이어서 진행된다 (chunk_id 기준 resume).

사용법: python scripts/build_prospectus_vectorstore.py
"""

import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.storage.prospectus_vectorstore import build_prospectus_vectorstore  # noqa: E402

DB_PATH = "data/processed/prospectus.db"
PROSPECTUS_ROOT = os.path.join("data", "raw", "prospectus")


def _find_pdf_dir() -> str:
    subdirs = [d for d in os.listdir(PROSPECTUS_ROOT) if os.path.isdir(os.path.join(PROSPECTUS_ROOT, d))]
    assert len(subdirs) == 1, f"예상 밖의 폴더 구조: {subdirs}"
    return os.path.join(PROSPECTUS_ROOT, subdirs[0])


def main():
    print("투자설명서 서술형 벡터DB 적재 시작 (문서 준비 → 임베딩)")
    _, stats = build_prospectus_vectorstore(db_path=DB_PATH, pdf_root=_find_pdf_dir())
    print(
        f"\n완료: 총 {stats.documents}개 문서 적재 / 위험표 추출 성공 {stats.funds_with_risk}개 펀드"
        f" / 실패 {stats.funds_missing_risk}개 펀드"
    )
    if stats.funds_missing_risk:
        print("실패 펀드는 prepare_prospectus_documents 로그로 확인 후 risk_extractor 패턴을 보강하세요.")


if __name__ == "__main__":
    main()
