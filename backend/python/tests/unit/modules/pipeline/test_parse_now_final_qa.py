"""Final QA probes for parse-now (notes/record-metadata-early/QA_B_FINAL.md)."""

from __future__ import annotations

import asyncio
import logging
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config.constants.arangodb import EventTypes
from app.modules.indexing import parse_now_marker as marker
from app.modules.pipeline.parse_now import (
    READY,
    ParseNow,
    optional_redis,
    parse_outcome,
)
from app.services.messaging.config import StreamMessage
from tests.unit.services.messaging.test_record_handler_parse_now import (
    Harness,
    _payload,
    _record,
)
from tests.unit.utils.test_fetch_record_resolution import (
    MODULE,
    _cached_record,
    _Docs,
    _fetch,
    _provider,
    _row,
    _stored,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _quick(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(f"{MODULE}._POLL_STEPS", (0.01,))
    monkeypatch.setattr(f"{MODULE}._POLL_THEN_EVERY", 0.01)
    yield
    patch.stopall()


@pytest.fixture
def stored_content():
    async def fake_get_record(vrid, mapping, *_a, **_kw) -> None:
        mapping[vrid] = _cached_record("rec-1")

    with patch(f"{MODULE}.get_record", side_effect=fake_get_record) as mock:
        yield mock


def _asker(wait: float) -> tuple[ParseNow, AsyncMock]:
    producer = MagicMock()
    producer.send_message = AsyncMock(return_value=True)
    patch(f"{MODULE}.parse_now_wait_seconds", AsyncMock(return_value=wait)).start()
    config = MagicMock()
    config.get_config = AsyncMock(return_value={})
    return ParseNow(config, logging.getLogger("qa"), producer), producer.send_message


# --- parse_outcome: a missing revision matches a missing revision, and only that ---------------------


@pytest.mark.parametrize(("doc", "expected"), [
    ({"indexingStatus": "AUTO_INDEX_OFF", "parseNowStatus": "EMPTY", "parseNowRev": None, "contentRev": None}, "EMPTY"),
    ({"indexingStatus": "AUTO_INDEX_OFF", "parseNowStatus": "EMPTY", "contentRev": None}, "EMPTY"),
    ({"indexingStatus": "AUTO_INDEX_OFF", "parseNowStatus": "EMPTY", "parseNowRev": None, "contentRev": "r2"}, None),
    ({"indexingStatus": "AUTO_INDEX_OFF", "parseNowStatus": "EMPTY", "parseNowRev": "r1", "contentRev": "r2"}, None),
    ({"indexingStatus": "AUTO_INDEX_OFF", "parseNowStatus": "FAILED", "parseNowRev": "r1", "contentRev": "r2"}, None),
    ({"indexingStatus": "AUTO_INDEX_OFF", "parseNowStatus": "STORED", "parseNowRev": None, "contentRev": None}, None),
    ({"indexingStatus": "AUTO_INDEX_OFF", "parseNowStatus": "STORED", "parseNowRev": "r1", "contentRev": "r1"}, READY),
    ({"indexingStatus": "PAUSED"}, "PAUSED"),
])
async def test_parse_outcome_revision_matching(doc, expected) -> None:
    assert parse_outcome(doc) == expected


# --- FAILED marker: ends the wait, does not block a republish -----------------------------------------


async def test_a_failed_marker_ends_the_wait_and_the_next_read_asks_again() -> None:
    failed = _row("AUTO_INDEX_OFF", parseNowStatus="FAILED", parseNowRev="rev-1")
    asker, send = _asker(wait=5)
    t0 = time.monotonic()
    first = await _fetch(
        ["rec-1"], graph_provider=_provider(_Docs({"rec-1": [_row("AUTO_INDEX_OFF"), failed]})), parse_now=asker,
    )
    assert time.monotonic() - t0 < 1.0
    assert first["not_indexed_records"]["rec-1"]["status"] == "FAILED"
    assert "preparing" not in first["not_indexed_records"]["rec-1"]
    # A later read of a record that still carries the FAILED marker asks again.
    second, send2 = _asker(wait=0.05)
    await _fetch(["rec-1"], graph_provider=_provider(_Docs({"rec-1": [failed]})), parse_now=second)
    assert send.await_count == 1 and send2.await_count == 1


async def test_a_stale_failed_marker_does_not_end_the_wait_of_a_run_just_asked_for(stored_content) -> None:
    stale = _row("FAILED", parseNowStatus="FAILED", parseNowRev="rev-1")
    docs = _Docs({"rec-1": [stale, stale, stale, _stored(stale)]})
    asker, send = _asker(wait=2)

    result = await _fetch(["rec-1"], graph_provider=_provider(docs), parse_now=asker)

    send.assert_awaited_once()
    assert result["ok"] is True


# --- blob read grace: a hung read of what became ready is cut near the budget ----------------------


async def test_a_hung_blob_read_after_ready_is_cut_near_the_budget(monkeypatch) -> None:
    monkeypatch.setattr(f"{MODULE}._FINISH_GRACE_SECONDS", 0.2)

    async def hang(*_a, **_k) -> None:
        await asyncio.sleep(30)

    docs = _Docs({"rec-1": [_row("QUEUED"), _stored(_row("QUEUED"))]})
    asker, _ = _asker(wait=0.2)
    with patch(f"{MODULE}.get_record", side_effect=hang):
        t0 = time.monotonic()
        result = await _fetch(["rec-1"], graph_provider=_provider(docs), parse_now=asker)
        elapsed = time.monotonic() - t0

    assert elapsed < 1.5
    assert result["ok"] is False and "being prepared" in result["error"]


# --- retries exhausted: marker only, never FAILED on the record ---------------------------------------


@pytest.mark.parametrize("status", ["AUTO_INDEX_OFF", "QUEUED", "NOT_STARTED", "IN_PROGRESS", "FAILED"])
async def test_an_abandoned_parse_now_event_writes_only_the_marker(status) -> None:
    harness = Harness(_record(indexingStatus=status))
    message = StreamMessage(eventType=EventTypes.REINDEX_RECORD.value, payload=_payload())

    await harness.handler.on_message_abandoned(message, reason="retries exhausted", attempts=5)

    writes = harness.writes()
    assert [w["parseNowStatus"] for w in writes] == ["FAILED"]
    assert all(set(w) == {"parseNowStatus", "parseNowRev", "parseNowReason"} for w in writes)
    harness.graph.compare_and_set_indexing_status.assert_not_awaited()
    harness.handler._trigger_next_queued_duplicate.assert_not_awaited()
    harness.handler._propagate_primary_failure_to_queued_duplicates.assert_not_awaited()


# --- STORED is never overwritten ------------------------------------------------------------------


@pytest.mark.parametrize("fields", [
    {"indexingStatus": "FAILED", "reason": "embedding failed"},
    {"indexingStatus": "QUEUED"},
    {"indexingStatus": "IN_PROGRESS"},
    {"indexingStatus": "EMPTY"},
])
@pytest.mark.parametrize("store_only", [True, False])
async def test_no_status_write_hides_stored_content(fields, store_only) -> None:
    marker.begin_run(marker.ParseNowRun(store_only=store_only, content_rev="r1", stored=True))
    try:
        written = marker.shape_status_write(fields)
    finally:
        marker.end_run()
    assert marker.PARSE_NOW_STATUS not in written


async def test_no_run_means_no_marker_and_no_leak() -> None:
    assert marker.shape_status_write({"indexingStatus": "FAILED"}) == {"indexingStatus": "FAILED"}
    assert marker.current_run() is None


# --- store-only: no record status field survives shaping --------------------------------------------


@pytest.mark.parametrize("status", [
    "QUEUED", "NOT_STARTED", "IN_PROGRESS", "COMPLETED", "FAILED", "EMPTY", "PAUSED", "SKIPPED", "AUTO_INDEX_OFF",
    "PASSWORD_PROTECTED", "FILE_TYPE_NOT_SUPPORTED", "ENABLE_MULTIMODAL_MODELS", "CONNECTOR_OFF", None,
])
async def test_store_only_never_writes_a_record_status(status) -> None:
    every = {
        "indexingStatus": status, "parsingStatus": status, "extractionStatus": status, "processingStartedAt": 1,
        "queuedAtTimestamp": 1, "isDirty": False, "lastIndexTimestamp": 1, "lastExtractionTimestamp": 1,
        "hasIndexedCopy": True, "extractionDeferred": True, "reason": "x", "retryNotBefore": 1,
        "contentRev": "r1", "virtualRecordId": "vr",
    }
    marker.begin_run(marker.ParseNowRun(store_only=True, content_rev="r1"))
    try:
        written = marker.shape_status_write(every)
    finally:
        marker.end_run()
    assert set(written) <= {"contentRev", "virtualRecordId", "parseNowStatus", "parseNowRev", "parseNowReason"}


async def test_a_stale_failed_marker_ends_the_wait_once_a_new_run_failed(stored_content) -> None:
    stale = _row("FAILED", parseNowStatus="FAILED", parseNowRev="rev-1")
    running = _row("IN_PROGRESS", parseNowStatus="IN_PROGRESS", parseNowRev="rev-1")
    docs = _Docs({"rec-1": [stale, running, stale]})
    asker, send = _asker(wait=2)

    result = await _fetch(["rec-1"], graph_provider=_provider(docs), parse_now=asker)

    send.assert_awaited_once()
    assert result["ok"] is False


async def test_a_redis_failure_runs_parse_now_without_dedup(caplog) -> None:
    config_service = MagicMock()
    config_service.get_redis_config = AsyncMock(side_effect=ConnectionError("down"))

    with caplog.at_level(logging.WARNING):
        assert await optional_redis(config_service, logging.getLogger("qa")) is None

    assert "Redis unavailable" in caplog.text and "down" in caplog.text


async def test_the_redis_handle_comes_from_the_services_own_config_service() -> None:
    # The container's redis_provider is bound to the base config_service (no key_value_store), so it raised TypeError in
    # the query service; this path must need nothing but get_redis_config().
    from app.config.configuration_service import ConfigurationService
    from app.containers.query import QueryAppContainer
    from app.services.redis.config import RedisConnectionConfig
    from app.services.redis.connection_provider_factory import (
        reset_redis_provider_registry,
    )

    assert QueryAppContainer.config_service.cls is ConfigurationService
    assert "key_value_store" in QueryAppContainer.config_service.kwargs  # the shape that broke the container's provider

    reset_redis_provider_registry()
    config_service = MagicMock()
    config_service.get_redis_config = AsyncMock(
        return_value=MagicMock(host="localhost", port=6379, password=None, db=0, tls=False)
    )
    try:
        provider = await optional_redis(config_service, logging.getLogger("qa"))
    finally:
        reset_redis_provider_registry()

    assert provider is not None and hasattr(provider, "get_client")
    assert RedisConnectionConfig.from_host_port  # signature used above


async def test_a_permanent_marker_ends_the_wait_and_is_not_asked_again() -> None:
    doc = {
        "indexingStatus": "AUTO_INDEX_OFF", "isDeleted": False, "contentRev": "r1",
        "parseNowStatus": "PASSWORD_PROTECTED", "parseNowRev": "r1",
    }
    assert parse_outcome(doc) == "PASSWORD_PROTECTED"
    assert await ParseNow(MagicMock(), logging.getLogger("qa")).assess(MagicMock(), doc) == (False, "PASSWORD_PROTECTED")
