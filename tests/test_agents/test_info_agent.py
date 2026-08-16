from src.agents.info_agent import _force_doc_search


class FakeSearchTool:
    def __init__(self, result):
        self.result = result

    def invoke(self, args):
        assert args["query"] == "연금계좌의 세금혜택"
        return self.result


def test_force_doc_search_converts_search_results_to_context(monkeypatch):
    monkeypatch.setattr(
        "src.agents.info_agent.search_pension_docs",
        FakeSearchTool([
            {
                "file_title": "연금계좌 세금혜택 안내",
                "section": "세액공제",
                "content": "연금계좌는 납입 시 세액공제 혜택이 있습니다.",
            }
        ]),
    )

    context = _force_doc_search("연금계좌의 세금혜택")

    assert context == [
        {
            "source": "연금계좌 세금혜택 안내 — 세액공제",
            "content": "연금계좌는 납입 시 세액공제 혜택이 있습니다.",
            "node": "info_agent",
        }
    ]


def test_force_doc_search_ignores_non_list_result(monkeypatch):
    monkeypatch.setattr("src.agents.info_agent.search_pension_docs", FakeSearchTool({"error": "failed"}))

    assert _force_doc_search("연금계좌의 세금혜택") == []


def test_force_doc_search_handles_search_error(monkeypatch):
    class BrokenSearchTool:
        def invoke(self, args):
            raise RuntimeError("search failed")

    monkeypatch.setattr("src.agents.info_agent.search_pension_docs", BrokenSearchTool())

    assert _force_doc_search("연금계좌의 세금혜택") == []
