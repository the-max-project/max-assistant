#!/usr/bin/env python3
"""
Native Dump Checkpointer Tool for Max Assistant.
Uses the LangGraph Neo4jSaver object model natively to unpack threads and message lineages.
"""

import os
import sys
import argparse
from neo4j import GraphDatabase
from langchain_neo4j.checkpoint import Neo4jSaver
from max_assistant.utils.checkpoints import dump_thread_transcript

# Safeguard check guardrail
if Neo4jSaver is None:
    print("\n🚨 Framework Discovery Error: Python cannot resolve the Neo4jSaver class path.")
    print("Please explicitly verify your active environment packages by running:")
    print("   python3 -c 'import langchain_neo4j; print(langchain_neo4j.__file__)'\n")
    sys.exit(1)

from local_config import init_environment

init_environment(True)

# Pull system environment parameters
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DB = os.getenv("NEO4J_DATABASE", "neo4j")


def list_active_threads():
    """Lists all distinct conversation threads ordered by the most recent activity first."""

    # Bypassing saver.list() limitation for global scans by querying unique thread IDs directly
    try:
        with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)) as db_driver:
            with db_driver.session(database=NEO4J_DB) as session:
                result = session.run("""
                MATCH (n:Thread)
                RETURN count(n) AS total_threads
                """)
                record = result.single()
                total_threads = record["total_threads"] if record else 0

                result = session.run("""
                MATCH (n:Thread)-[r:HAS_BRANCH]->(b:Branch) 
                where b.name="main" 
                RETURN n.thread_id as thread_id, b.created_at as created_at 
                ORDER BY b.created_at DESC LIMIT 50
                """)
                threads = [(record["thread_id"], record["created_at"].isoformat()) for record in result if record["thread_id"]]
                num_threads = len(threads)
    except Exception as e:
        print(f"  [!] Failed to query unique thread IDs from Neo4j: {e}")
        return

    if not threads:
        print("  [!] No thread records found")
        return

    print("\n" + "=" * 80)
    print(f"{'ACTIVE CONVERSATION THREADS':^80}")
    print(f"Listing {num_threads} most recent of {total_threads} total threads")
    print("=" * 80 + "\n")
    print("Thread_Id                             Created at")
    print("------------------------------------  ----------")
    for thread_id, created_at in threads:
        print(f"{thread_id}  {created_at}")


def main():
    parser = argparse.ArgumentParser(description="Query and inspect LangGraph checkpoints natively using Neo4jSaver.")
    parser.add_argument("thread_id", nargs="?", help="Optional Target Thread ID to view conversation logs.")
    args = parser.parse_args()

    try:
        if hasattr(Neo4jSaver, 'from_conn_string'):
            with Neo4jSaver.from_conn_string(
                    uri=NEO4J_URI,
                    user=NEO4J_USER,
                    password=NEO4J_PASSWORD,
                    database=NEO4J_DB
            ) as saver:
                if args.thread_id:
                    dump_thread_transcript(saver, args.thread_id)
                else:
                    list_active_threads()
        else:
            # Alternate instance routing matching standard langchain_neo4j constructor layouts
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            saver = Neo4jSaver(driver=driver, database=NEO4J_DB)

            if args.thread_id:
                dump_thread_transcript(saver, args.thread_id)
            else:
                list_active_threads()

    except Exception as e:
        print(f"🚨 Failed to execute native Neo4jSaver operation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()