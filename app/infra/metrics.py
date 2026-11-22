"""Process-local service counters, exposed in Prometheus text format."""

from threading import Lock

_lock = Lock()
_values: dict[str, int | float] = {
    "astronomy_search_lexical_total": 0,
    "astronomy_search_hybrid_total": 0,
    "astronomy_search_cache_hits_total": 0,
    "astronomy_embedding_requests_total": 0,
    "astronomy_embedding_errors_total": 0,
    "astronomy_embedding_request_seconds_total": 0.0,
    "astronomy_assistant_errors_total": 0,
}


def count(metric: str, amount: int | float = 1) -> None:
    with _lock:
        _values[metric] += amount


def render() -> str:
    with _lock:
        return "".join(f"# TYPE {name} counter\n{name} {value}\n" for name, value in _values.items())
