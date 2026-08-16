from langchain_core.messages import AIMessage, HumanMessage

from src.agents.context import format_conversation_history, history_to_messages


def test_format_conversation_history_adds_previous_turns():
    history = [
        {"question": "58세인데 크게 잃지 않을 상품 추천해줘", "answer": "계좌유형과 투자기간을 알려주세요."},
    ]

    prompt = format_conversation_history(history)

    assert "사용자: 58세인데 크게 잃지 않을 상품 추천해줘" in prompt
    assert "답변: 계좌유형과 투자기간을 알려주세요." in prompt


def test_history_to_messages_adds_previous_turns():
    messages = history_to_messages([
        {"question": "상품 추천해줘", "answer": "계좌유형을 알려주세요."},
    ])

    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[0].content == "상품 추천해줘"


def test_format_conversation_history_empty():
    assert format_conversation_history([]) == ""
