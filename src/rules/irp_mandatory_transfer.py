"""퇴직 시 IRP 의무이전 규칙.

근로자가 퇴직하면 퇴직급여는 원칙적으로 IRP 계좌로 이전된다. 법정 예외사유에
해당할 때만 개인(예금)계좌로 직접 받을 수 있다.

## 왜 규칙으로 두는가

이 규칙이 없던 동안 실측에서 두 건이 어긋났다:
  - no.361 "DC형 계좌는 이직할 때마다 새로 생기나요?" -> "네, 새롭게 개설되는
    경우가 많습니다"라고 답했다. 실제로는 퇴직 시 DC 적립금이 IRP로 이전되는
    것이지 DC 계좌가 새로 생기는 구조가 아니다.
  - no.17 "퇴사하면 IRP를 반드시 만들어야 하나요?" -> "DC 퇴직금은 나이와
    상관없이 반드시 IRP로 이전해야 한다"고 단정했다. 55세 이후 퇴직은 예외다.

두 오답 모두 "원칙(의무이전)"과 "예외(직접수령)" 중 한쪽만 말해서 생겼다.
근거: doc "퇴직연금제도 기본 — 개인형 퇴직연금제도(IRP)",
      doc "퇴직연금 사무담당자 업무 매뉴얼 — 업무 체크 포인트".
"""

from __future__ import annotations

# 의무이전 예외사유 — 이 중 하나라도 해당하면 퇴직급여를 개인계좌로 직접 받을 수 있다.
# (해당해도 IRP로 받는 것을 선택할 수 있다 — 예외는 "금지"가 아니라 "의무 면제"다.)
IRP_MANDATORY_TRANSFER_EXCEPTIONS: tuple[str, ...] = (
    "55세 이후에 퇴직하는 경우",
    "퇴직급여액이 300만원 이하인 경우",
    "가입자가 사망하거나 해외이주하는 경우",
    "퇴직연금 담보대출 상환 등 대통령령이 정하는 사유에 해당하는 경우",
)

# 예외사유로 개인계좌로 받은 뒤에도, 이 기간 안에는 IRP로 납입해 과세이연을 유지할 수 있다.
IRP_POST_RECEIPT_DEPOSIT_DAYS = 60

# 퇴직급여 지급 기한 (사용자 의무).
SEVERANCE_PAYMENT_DEADLINE_DAYS = 14

# 소액 예외 기준액 (원).
IRP_SMALL_AMOUNT_EXCEPTION = 3_000_000


def is_mandatory_transfer_exempt(age: int | None = None, severance_amount: int | None = None) -> bool | None:
    """확인된 조건만으로 의무이전 예외에 해당하는지 판정한다.

    판단에 필요한 값이 하나도 없으면 None을 돌려준다 — 모르는 조건을 "예외 아님"으로
    단정하면 "무조건 IRP로 이전해야 한다"는 실측 오답(no.17)이 그대로 재현된다.
    """
    if age is None and severance_amount is None:
        return None
    if age is not None and age >= 55:
        return True
    if severance_amount is not None and severance_amount <= IRP_SMALL_AMOUNT_EXCEPTION:
        return True
    return False
