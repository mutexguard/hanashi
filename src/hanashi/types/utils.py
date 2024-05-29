import uuid as _uuid


def uuid() -> str:
    return _uuid.uuid4().hex
