"""Smoke for a clean deployment with real Redis and empty LLM credentials."""
import argparse
import json
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def request(base, path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = Request(base + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=15) as response:
            return response.status, response.headers, response.read()
    except HTTPError as exc:
        with exc:
            return exc.code, exc.headers, exc.read()


def get_json(base, path):
    status, _, body = request(base, path)
    assert status == 200, (path, status)
    return json.loads(body)


def run(base, peer, expected_mode):
    for path in ("/health", "/docs", "/openapi.json"):
        assert request(base, path)[0] == 200, path
    readiness = get_json(base, "/ready")
    assert readiness["catalogue"] == "ready"
    if expected_mode == "lexical":
        assert readiness["status"] == "degraded"
    elif expected_mode == "hybrid":
        assert readiness["semantic"] == "ready"
    assert b"astronomy_search_lexical_total" in request(base, "/metrics")[2]
    assert get_json(base, "/api/v1/terms")["total"] == 501
    if peer != base:
        assert get_json(peer, "/ready")["catalogue"] == "ready"
        assert get_json(peer, "/api/v1/terms")["total"] == 501
    article = get_json(base, "/api/v1/articles/article:exoplanet-detection")
    assert {"ru", "en"} == set(article["locales"])
    assert {"transit", "radial-velocity"} <= {section["id"] for section in article["sections"]}

    review = "/api/v1/articles/article:oae-document-earthasaplanet_mini_review"
    for filename in ("source.pdf", "translation.pdf"):
        status, headers, pdf = request(base, f"{review}/{filename}")
        assert status == 200 and headers.get_content_type() == "application/pdf", (filename, status)
        assert pdf.startswith(b"%PDF-"), filename

    for locale, query in (("ru", "экзопланета"), ("en", "exoplanet")):
        params = urlencode({"q": query, "locale": locale, "page_size": 5})
        result = get_json(base, "/api/v1/search?" + params)
        assert any(item["record_id"] == "term:106" for item in result["items"]), result
        if expected_mode:
            assert result["mode"] == expected_mode, result["mode"]
        print(locale, result["mode"], "term:106")

    illustration = get_json(base, "/api/v1/terms/106")["illustration"]
    status, headers, cover = request(base, illustration["url"])
    assert status == 200 and headers.get_content_type() == "image/webp"
    assert cover[:4] == b"RIFF" and cover[8:12] == b"WEBP"
    assert int.from_bytes(cover[4:8], "little") + 8 == len(cover)
    assert "immutable" in headers["Cache-Control"]
    print("WebP bytes", len(cover))

    original = get_json(base, "/api/v1/media/diagram:zF97Dc8Sp10")
    status, headers, diagram = request(base, original["file_url_en"])
    assert status == 200 and diagram.startswith(b"\x89PNG\r\n\x1a\n")
    assert headers["ETag"] == f'"{original["sha256"]}"'
    print("Original PNG bytes", len(diagram))

    path = "/api/v1/assistant/answer"
    body = {"question": "Что такое экзопланета?", "locale": "ru"}
    status, _, payload = request(base, path, body)
    assert status == 503 and json.loads(payload)["code"] == "assistant_unavailable"
    status, headers, payload = request(peer, path, body)
    assert status == 429 and json.loads(payload)["code"] == "rate_limited"
    assert 1 <= int(headers["Retry-After"]) <= 10
    print(status, "Retry-After", headers["Retry-After"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("--peer", help="Second API process sharing the same Redis")
    parser.add_argument("--search-mode", choices=("hybrid", "lexical"))
    args = parser.parse_args()
    run(args.base.rstrip("/"), (args.peer or args.base).rstrip("/"), args.search_mode)
