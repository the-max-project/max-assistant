"""Print checkpoint histories without owning the underlying database connection."""

import textwrap


def _print_transcript(thread_id, history):
    print("\n" + "=" * 80)
    print(f"{f'TRANSCRIPT FOR THREAD: {thread_id}':^80}")
    print("=" * 80 + "\n")

    if not history:
        print(f"  [!] No active checkpoints found matching thread ID: {thread_id}")
        return

    print(f"Total Checkpoints Found: {len(history)}\n")
    for idx, cp_tuple in enumerate(reversed(history), 1):
        cp_id = cp_tuple.config.get("configurable", {}).get("checkpoint_id")
        metadata = cp_tuple.metadata or {}
        print(f" [{idx}] Checkpoint ID : {cp_id}")
        print(f"     Graph Workflow: Source={metadata.get('source')} | Step={metadata.get('step')}")
        print("     Messages Transcript:")
        messages = cp_tuple.checkpoint.get("channel_values", {}).get("messages", [])
        if messages:
            for msg in messages:
                raw_msg = getattr(msg, "type", type(msg).__name__.lower())
                msg_type = raw_msg.decode("utf8", errors="replace") if isinstance(raw_msg, (bytes, bytearray)) else str(raw_msg)
                if "human" in msg_type:
                    label = "[HUMAN]"
                elif "ai" in msg_type:
                    label = "[AI]"
                elif "tool" in msg_type:
                    label = "[TOOL]"
                else:
                    label = f"[{msg_type.upper()}]"
                content = getattr(msg, "content", str(msg))
                if not isinstance(content, str):
                    content = str(content)
                prefix = f"         {label:<9}: "
                indented_content = textwrap.indent(content, " " * len(prefix))[len(prefix):]
                print(f"{prefix}{indented_content}")
        else:
            print("         (No active message frames in this state channel step)")
        print("-" * 80)


def dump_thread_transcript(saver, thread_id):
    """Print a synchronous saver's checkpoints from oldest to newest."""
    config = {"configurable": {"thread_id": thread_id}}
    _print_transcript(thread_id, list(saver.list(config)))


async def async_dump_thread_transcript(saver, thread_id, *, latest_only=False):
    """Print an asynchronous saver's checkpoints, optionally only the newest."""
    config = {"configurable": {"thread_id": thread_id}}
    if latest_only:
        checkpoint = await saver.aget_tuple(config)
        history = [checkpoint] if checkpoint else []
    else:
        history = [checkpoint async for checkpoint in saver.alist(config)]
    _print_transcript(thread_id, history)