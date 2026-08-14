import openpyxl

from src.storage.docs_vectorstore import prepare_documents

HEADER = [
    "파일번호", "파일제목", "파일형식", "카테고리(복수 가능)", "저장방식", "section",
    "원문위치", "chunk_id", "chunk_text", "표 포함", "표/구조화 텍스트", "특이사항", "검수상태",
]


def _make_xlsm(path):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "파싱 결과_검수본"
    for _ in range(4):
        ws.append([None])
    ws.append(HEADER)
    ws.append([
        1, "테스트 문서", "pdf", "1. 연금기초", "RAG", "테스트 섹션",
        "p.1", "doc1_chunk01", "테스트 청크 본문입니다.", "N", None, None, "검수완료",
    ])
    ws.append([
        1, "테스트 문서", "pdf", "1. 연금기초", "RAG", "표가 있는 섹션",
        "p.2", "doc1_chunk02", "표 설명 문단입니다.", "Y",
        "| 구분 | 값 |\n|---|---|\n| A | 1 |", None, "검수완료",
    ])
    ws.append([
        2, "빈 청크 문서", "pdf", "1. 연금기초", "RAG", "빈 섹션",
        "p.1", "doc2_chunk01", "", "N", None, None, "검수완료",
    ])
    wb.save(path)


def test_prepare_documents_builds_documents_with_metadata(tmp_path):
    xlsm_path = tmp_path / "docs.xlsm"
    _make_xlsm(xlsm_path)

    docs = prepare_documents(str(xlsm_path), sheet_name="파싱 결과_검수본")

    assert len(docs) == 2  # 빈 chunk_text 행은 제외되어야 함
    assert docs[0].metadata["chunk_id"] == "doc1_chunk01"
    assert docs[0].page_content == "테스트 청크 본문입니다."

    # 표/구조화 텍스트가 있으면 본문에 합쳐져야 함
    assert "표 설명 문단입니다." in docs[1].page_content
    assert "| 구분 | 값 |" in docs[1].page_content
    assert docs[1].metadata["has_table"] == "Y"


def test_prepare_documents_skips_rows_without_chunk_id_or_text(tmp_path):
    xlsm_path = tmp_path / "docs.xlsm"
    _make_xlsm(xlsm_path)

    docs = prepare_documents(str(xlsm_path), sheet_name="파싱 결과_검수본")
    chunk_ids = {d.metadata["chunk_id"] for d in docs}

    assert "doc2_chunk01" not in chunk_ids
