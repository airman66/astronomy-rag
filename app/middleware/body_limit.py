from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.api.errors import error
from app.api.paths import API_PREFIX, ASSISTANT_ANSWER_PATH

MAX_ANSWER_BODY = 16 * 1024


class AssistantBodyLimitMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != API_PREFIX + ASSISTANT_ANSWER_PATH:
            await self.app(scope, receive, send)
            return

        lengths = [value for name, value in scope["headers"] if name.lower() == b"content-length"]
        if lengths:
            if len(lengths) != 1 or not lengths[0].isdigit():
                await error(400, "invalid_content_length", "Некорректный Content-Length")(
                    scope, receive, send
                )
                return
            if int(lengths[0]) > MAX_ANSWER_BODY:
                await error(413, "body_too_large", "Тело запроса превышает 16 КиБ")(scope, receive, send)
                return

        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                await error(400, "invalid_request_body", "Не удалось прочитать тело запроса")(
                    scope, receive, send
                )
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > MAX_ANSWER_BODY:
                await error(413, "body_too_large", "Тело запроса превышает 16 КиБ")(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        payload = bytes(body)
        delivered = False

        async def replay_receive() -> Message:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": payload, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)
