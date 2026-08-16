"""투자설명서 서술형 텍스트(투자목적/투자전략/투자위험)를 Chroma 벡터DB로 적재한다.

"이 펀드의 투자전략·위험이 뭐예요" 같은 단일 상품 설명 질의(대회 대주제 2의 첫 축)는
구조화 DB의 숫자 필드만으로 답할 수 없다 — 서술 원문이 근거로 필요하다. 문서 구성:

  - 투자목적/투자전략: fund_master의 검수 완료된 값 (사람이 원문 대조한 텍스트를 재사용)
  - 투자위험: PDF 요약 위험표에서 추출 (src/parsing/risk_extractor.py)

docs.zip 벡터DB(pension_docs)와 별도 컬렉션·디렉터리를 쓴다. 임베딩 한도(600자) 분할과
재개(resume) 로직은 docs_vectorstore와 동일한 방식을 재사용한다.
"""

import os
import unicodedata
from dataclasses import dataclass

from langchain_core.documents import Document

from src.storage.docs_vectorstore import _RateLimitedEmbeddings, _split_long_documents

DEFAULT_PROSPECTUS_CHROMA_DIR = "data/processed/chroma_prospectus"
DEFAULT_PROSPECTUS_COLLECTION = "prospectus_text"


@dataclass
class LoadStats:
    documents: int = 0
    funds_with_risk: int = 0
    funds_missing_risk: int = 0


def prepare_prospectus_documents(db_path: str, pdf_root: str) -> tuple[list[Document], LoadStats]:
    """DB의 투자목적/전략 + PDF의 위험 요약으로 임베딩 대상 Document를 만든다. 임베딩 호출 없음."""
    import sqlite3

    from src.parsing.risk_extractor import extract_risk_summary_from_pdf

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT product_code, fund_name, investment_objective, investment_strategy FROM fund_master"
    ).fetchall()
    conn.close()

    stats = LoadStats()
    documents: list[Document] = []
    for row in rows:
        code = row["product_code"]
        fund_name = unicodedata.normalize("NFC", row["fund_name"] or "")

        sections = []
        if row["investment_objective"]:
            sections.append(("투자목적", row["investment_objective"], "검수 DB"))
        if row["investment_strategy"]:
            sections.append(("투자전략", row["investment_strategy"], "검수 DB"))

        code_dir = os.path.join(pdf_root, code)
        pdfs = [f for f in os.listdir(code_dir) if f.lower().endswith(".pdf")] if os.path.isdir(code_dir) else []
        risk_text = extract_risk_summary_from_pdf(os.path.join(code_dir, pdfs[0])) if pdfs else None
        if risk_text:
            sections.append(("투자위험", risk_text, "투자설명서 요약 위험표"))
            stats.funds_with_risk += 1
        else:
            stats.funds_missing_risk += 1

        for section, text, origin in sections:
            documents.append(
                Document(
                    page_content=f"[{fund_name} — {section}]\n{text}",
                    metadata={
                        "chunk_id": f"{code}_{section}",
                        "product_code": code,
                        "fund_name": fund_name,
                        "section": section,
                        "origin": origin,
                    },
                )
            )

    documents = _split_long_documents(documents)
    stats.documents = len(documents)
    return documents, stats


def build_prospectus_vectorstore(
    db_path: str,
    pdf_root: str,
    persist_dir: str = DEFAULT_PROSPECTUS_CHROMA_DIR,
    collection_name: str = DEFAULT_PROSPECTUS_COLLECTION,
    request_delay_seconds: float = 1.5,
    batch_size: int = 20,
):
    """Document를 준비하고 CLOVA Studio로 임베딩하여 Chroma에 적재한다. API 키 필요.

    docs_vectorstore.build_vectorstore와 동일하게 rate limit 지연 + chunk_id 기준
    재개(resume)를 지원한다 — 중간에 실패해도 재실행하면 이어서 진행된다.
    """
    from langchain_chroma import Chroma

    from src.agents.llm import get_embeddings

    documents, stats = prepare_prospectus_documents(db_path, pdf_root)
    embeddings = _RateLimitedEmbeddings(get_embeddings(), delay_seconds=request_delay_seconds)

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )

    existing_ids = set(vectorstore.get(include=[])["ids"])
    documents = [d for d in documents if d.metadata["chunk_id"] not in existing_ids]
    if existing_ids:
        print(f"  이미 적재된 {len(existing_ids)}건 건너뜀, 남은 {len(documents)}건 진행")

    total = len(documents)
    for start in range(0, total, batch_size):
        batch = documents[start:start + batch_size]
        ids = [doc.metadata["chunk_id"] for doc in batch]
        vectorstore.add_documents(documents=batch, ids=ids)
        print(f"  적재 진행: {min(start + batch_size, total)}/{total}")

    stats.documents = vectorstore._collection.count()
    return vectorstore, stats
