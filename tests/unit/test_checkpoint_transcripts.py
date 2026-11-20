import asyncio
from importlib import import_module
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def checkpoints():
    return import_module("max_assistant.utils.checkpoints")


def checkpoint(checkpoint_id, messages, metadata=None):
    return SimpleNamespace(
        config={"configurable": {"checkpoint_id": checkpoint_id}},
        metadata=metadata,
        checkpoint={"channel_values": {"messages": messages}},
    )


def message(kind, content):
    return SimpleNamespace(type=kind, content=content)


class SyncSaver:
    def __init__(self, history):
        self.history = history
        self.list = MagicMock(side_effect=lambda config: iter(self.history))


class AsyncSaver:
    def __init__(self, history):
        self.history = history
        self.alist = MagicMock(side_effect=self._list)
        self.aget_tuple = AsyncMock(return_value=history[0] if history else None)

    async def _list(self, config, *, limit=None):
        for item in self.history[:limit]:
            yield item


@pytest.mark.asyncio
async def test_sync_and_async_dump_share_format_and_reverse_history(checkpoints, capsys):
    history = [
        checkpoint("new", [message(b"ai", "first line\nsecond line"),
                           message("tool", "tool result"),
                           message("system", "instructions")], {"source": "loop", "step": 2}),
        checkpoint("old", [message("human", "hello")], {"source": "input", "step": 1}),
    ]
    sync_saver = SyncSaver(history)
    async_saver = AsyncSaver(history)
    config = {"configurable": {"thread_id": "thread-123"}}

    checkpoints.dump_thread_transcript(sync_saver, "thread-123")
    sync_output = capsys.readouterr().out
    await checkpoints.async_dump_thread_transcript(async_saver, "thread-123")
    async_output = capsys.readouterr().out

    sync_saver.list.assert_called_once_with(config)
    async_saver.alist.assert_called_once_with(config)
    assert sync_output == async_output
    assert "TRANSCRIPT FOR THREAD: thread-123" in sync_output
    assert "Total Checkpoints Found: 2" in sync_output
    assert sync_output.index("Checkpoint ID : old") < sync_output.index("Checkpoint ID : new")
    assert "Source=input | Step=1" in sync_output
    assert "Source=loop | Step=2" in sync_output
    assert "[HUMAN]" in sync_output and "hello" in sync_output
    assert "[AI]" in sync_output and "first line" in sync_output
    assert "[TOOL]" in sync_output and "tool result" in sync_output
    assert "[SYSTEM]" in sync_output and "instructions" in sync_output
    first_line, second_line = (line for line in sync_output.splitlines()
                               if "first line" in line or "second line" in line)
    assert first_line.index("first line") == second_line.index("second line")


@pytest.mark.asyncio
async def test_async_dump_can_select_latest_checkpoint(checkpoints, capsys):
    saver = AsyncSaver([
        checkpoint("new", [message("human", "complete conversation")]),
        checkpoint("old", [message("human", "older snapshot")]),
    ])

    await checkpoints.async_dump_thread_transcript(saver, "thread", latest_only=True)

    output = capsys.readouterr().out
    saver.aget_tuple.assert_awaited_once_with({"configurable": {"thread_id": "thread"}})
    saver.alist.assert_not_called()
    assert "Checkpoint ID : new" in output
    assert "complete conversation" in output
    assert "older snapshot" not in output


@pytest.mark.asyncio
async def test_async_dump_latest_with_no_checkpoint(checkpoints, capsys):
    await checkpoints.async_dump_thread_transcript(AsyncSaver([]), "empty", latest_only=True)
    assert "No active checkpoints found matching thread ID: empty" in capsys.readouterr().out


@pytest.mark.asyncio
@pytest.mark.parametrize("history, expected", [
    ([], "No active checkpoints found matching thread ID: empty"),
    ([checkpoint("empty-cp", [], None)], "No active message frames in this state channel step"),
])
async def test_empty_history_and_messages_match_between_dumpers(checkpoints, capsys, history, expected):
    checkpoints.dump_thread_transcript(SyncSaver(history), "empty")
    sync_output = capsys.readouterr().out
    await checkpoints.async_dump_thread_transcript(AsyncSaver(history), "empty")
    assert capsys.readouterr().out == sync_output
    assert expected in sync_output


@pytest.mark.asyncio
async def test_dumpers_propagate_saver_errors(checkpoints):
    sync_saver = SimpleNamespace(list=MagicMock(side_effect=RuntimeError("read failed")))

    async def failing_list(config):
        raise RuntimeError("read failed")
        yield

    with pytest.raises(RuntimeError, match="read failed"):
        checkpoints.dump_thread_transcript(sync_saver, "thread")
    with pytest.raises(RuntimeError, match="read failed"):
        await checkpoints.async_dump_thread_transcript(SimpleNamespace(alist=failing_list), "thread")


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled, dump_fails", [(False, False), (True, False), (True, True)])
async def test_runner_cleans_up_after_agent_failure(monkeypatch, enabled, dump_fails, capsys):
    from tests.function import test_runner

    ready = asyncio.Event()
    ready.set()
    scheduler = SimpleNamespace(start=AsyncMock(), stop=AsyncMock())
    client = SimpleNamespace(close=AsyncMock(), driver=object())
    services = SimpleNamespace(time_scheduler=scheduler, db_client=client,
                               llm_ready_event=ready, reasoning_engine=object())
    agent = MagicMock()
    agent.get_thread_id.return_value = "thread-123"
    agent.ainvoke = AsyncMock(side_effect=RuntimeError("agent failed"))
    monkeypatch.setattr(test_runner.AppServices, "create", AsyncMock(return_value=services))
    monkeypatch.setattr(test_runner, "Agent", MagicMock(return_value=agent))
    monkeypatch.setattr(test_runner, "PersonTools", lambda db: SimpleNamespace(
        get_user_info_internal=AsyncMock(return_value={"user": {"id": "user-1"}})))
    monkeypatch.setattr(test_runner.session_manager, "register", MagicMock())
    monkeypatch.setattr(test_runner.session_manager, "unregister", MagicMock())
    monkeypatch.setattr(test_runner, "_ACTIVE_MODEL", None)
    saver = object()
    saver_factory = MagicMock(return_value=saver)
    dump = AsyncMock(side_effect=RuntimeError("dump failed") if dump_fails else None)
    monkeypatch.setattr(test_runner, "AsyncNeo4jSaver", saver_factory)
    monkeypatch.setattr(test_runner, "async_dump_thread_transcript", dump)
    request = SimpleNamespace(node=SimpleNamespace(name="scenario"),
                              config=SimpleNamespace(getoption=MagicMock(return_value=enabled)))
    results = {}

    with pytest.raises(RuntimeError, match="agent failed"):
        await test_runner.execute_scenario_workflow(
            "alice", [{"user_input": "hello"}], "model", request, results)

    scheduler.stop.assert_awaited_once()
    test_runner.session_manager.unregister.assert_called_once_with("user-1", agent)
    client.close.assert_awaited_once()
    assert results["success"] is False
    assert results["thread_id"] == "thread-123"
    request.config.getoption.assert_called_once_with("--dump-checkpoints")
    if enabled:
        saver_factory.assert_called_once_with(driver=client.driver)
        dump.assert_awaited_once_with(saver, "thread-123", latest_only=True)
        if dump_fails:
            assert "dump failed" in capsys.readouterr().out
    else:
        saver_factory.assert_not_called()
        dump.assert_not_awaited()