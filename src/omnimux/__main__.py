import argparse
import asyncio
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import aiosqlite
import uvicorn

from omnimux.config import settings
from omnimux.services.cache import ImageCache
from omnimux.services.db import (
    create_client_key,
    export_configuration,
    init_db,
    list_client_keys,
    revoke_client_key,
)
from omnimux.services.model_discovery import (
    list_configured_models,
    run_add_model_wizard,
)

OMNIMUX_DIR = Path.home() / ".omnimux"
PID_FILE = OMNIMUX_DIR / ".omnimux.pid"
LOG_FILE = OMNIMUX_DIR / "omnimux.log"
PORT_FILE = OMNIMUX_DIR / ".omnimux.port"


async def _get_db() -> aiosqlite.Connection:
    db = await aiosqlite.connect(settings.sqlite_db_path)
    await init_db(db)
    return db


def handle_start(args: Any) -> None:
    host = getattr(args, "host", "0.0.0.0")
    port = getattr(args, "port", settings.port)
    detach = getattr(args, "detach", False)

    if detach:
        if PID_FILE.exists():
            try:
                old_pid = int(PID_FILE.read_text(encoding="utf-8").strip())
                os.kill(old_pid, 0)
                print(
                    f"Omnimux proxy is already running (PID: {old_pid}). Run 'omnimux stop' first."
                )
                return
            except (OSError, ValueError):
                PID_FILE.unlink(missing_ok=True)

        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as log_out:
            proc = subprocess.Popen(
                [sys.executable, "-m", "omnimux", "start", "--host", host, "--port", str(port)],
                stdout=log_out,
                stderr=log_out,
                start_new_session=True,
            )
        PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(proc.pid), encoding="utf-8")
        PORT_FILE.parent.mkdir(parents=True, exist_ok=True)
        PORT_FILE.write_text(str(port), encoding="utf-8")
        print(f"Omnimux proxy started in detached mode (PID: {proc.pid}). Logs: {LOG_FILE}")
        return

    uvicorn.run("omnimux.app:create_app", host=host, port=port, factory=True)


def handle_stop(args: Any = None) -> None:
    if not PID_FILE.exists():
        print("No active Omnimux PID file found.")
        return
    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Stopped Omnimux proxy (PID: {pid}).")
    except ProcessLookupError:
        print("Process not found. Cleaning up stale PID file.")
    except Exception as e:
        print(f"Error stopping proxy: {e}")
    finally:
        PID_FILE.unlink(missing_ok=True)
        PORT_FILE.unlink(missing_ok=True)


def handle_status(args: Any = None) -> None:
    if not PID_FILE.exists():
        print("Omnimux proxy is not running (no active PID file).")
        return

    try:
        pid = int(PID_FILE.read_text(encoding="utf-8").strip())
        os.kill(pid, 0)
    except (OSError, ValueError, ProcessLookupError):
        print("Omnimux proxy is not running (stale PID file found). Cleaning up.")
        PID_FILE.unlink(missing_ok=True)
        PORT_FILE.unlink(missing_ok=True)
        return

    port = getattr(args, "port", None)
    if not port and PORT_FILE.exists():
        try:
            port = int(PORT_FILE.read_text(encoding="utf-8").strip())
        except ValueError:
            port = settings.port
    if not port:
        port = settings.port

    import socket

    health = "Active"
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", port))
        health = "Healthy"
    except OSError:
        pass
    finally:
        s.close()

    print(
        f"Omnimux proxy is running (PID: {pid}, Port: {port}, Status: {health}). Logs: {LOG_FILE}"
    )


async def _handle_keys(args: Any) -> None:
    db = await _get_db()
    try:
        sub = getattr(args, "keys_action", None)
        if sub == "create":
            name = getattr(args, "name", "default") or "default"
            raw_key, _info = await create_client_key(db, name)
            print(f"\nCreated client API key for '{name}':")
            print(f"\033[1;32m{raw_key}\033[0m")
            print("Keep this token safe; it will not be shown again.")
            print(f'Use it with Claude Code: export ANTHROPIC_API_KEY="{raw_key}"\n')
        elif sub == "revoke":
            identifier = getattr(args, "identifier", "")
            success = await revoke_client_key(db, identifier)
            if success:
                print(f"Key '{identifier}' successfully revoked.")
            else:
                print(f"Key '{identifier}' not found or already revoked.")
        else:
            keys = await list_client_keys(db)
            if not keys:
                print("\nNo client keys found. Run 'omnimux keys create [name]' to generate one.\n")
                return
            print(f"\n{'ID':<5} {'Name':<20} {'Prefix':<16} {'Status':<10} {'Created'}")
            print("-" * 70)
            for k in keys:
                status_color = "\033[32m" if k["status"] == "active" else "\033[31m"
                print(
                    f"{k['id']:<5} {k['name']:<20} {k['prefix']:<16} "
                    f"{status_color}{k['status']:<10}\033[0m {k['created_at']}"
                )
            print()
    finally:
        await db.close()


async def _handle_model(args: Any) -> None:
    db = await _get_db()
    try:
        sub = getattr(args, "model_action", None)
        if sub == "add":
            await run_add_model_wizard(db)
        else:
            await list_configured_models(db)
    finally:
        await db.close()


async def _handle_export(args: Any) -> None:
    db = await _get_db()
    try:
        path = getattr(args, "path", None) or "omnimux_export.json"
        await export_configuration(db, path)
        print(f"Exported configuration to {path}")
    finally:
        await db.close()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:\t  %(name)s - %(message)s")

    parser = argparse.ArgumentParser(prog="omnimux", description="Omnimux LLM Proxy and Router CLI")
    subparsers = parser.add_subparsers(dest="command")

    # start subcommand
    start_parser = subparsers.add_parser("start", help="Start the Omnimux proxy server")
    start_parser.add_argument("--port", type=int, default=settings.port, help="Port to run on")
    start_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    start_parser.add_argument(
        "-d", "--detach", action="store_true", help="Run proxy in background/detached mode"
    )

    # stop subcommand
    subparsers.add_parser("stop", help="Stop detached Omnimux proxy server")

    # status subcommand
    status_parser = subparsers.add_parser("status", help="Inspect status of running Omnimux proxy")
    status_parser.add_argument("--port", type=int, default=None, help="Port to check")

    # model subcommand
    model_parser = subparsers.add_parser("model", help="Manage upstream models and platforms")
    model_sub = model_parser.add_subparsers(dest="model_action")
    model_sub.add_parser("add", help="Interactively add an upstream platform and models")
    model_sub.add_parser("list", help="List all configured models")

    # keys subcommand
    keys_parser = subparsers.add_parser("keys", help="Manage client API keys")
    keys_sub = keys_parser.add_subparsers(dest="keys_action")
    create_key_p = keys_sub.add_parser("create", help="Create a new client key")
    create_key_p.add_argument("name", nargs="?", default="default", help="Key name or label")
    keys_sub.add_parser("list", help="List all client keys")
    revoke_key_p = keys_sub.add_parser("revoke", help="Revoke a client key")
    revoke_key_p.add_argument("identifier", help="Key ID or prefix to revoke")

    # export subcommand
    export_parser = subparsers.add_parser("export", help="Export configuration to JSON")
    export_parser.add_argument(
        "path", nargs="?", default="omnimux_export.json", help="Destination file path"
    )

    # cache subcommand
    cache_parser = subparsers.add_parser("cache", help="Manage vision cache")
    cache_parser.add_argument("--clear", action="store_true", help="Clear cache")

    # Top-level backwards compatibility options
    parser.add_argument("--host", type=str, default="0.0.0.0", help=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=settings.port, help=argparse.SUPPRESS)
    parser.add_argument("-d", "--detach", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--clear-cache", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.clear_cache or (args.command == "cache" and getattr(args, "clear", False)):
        ImageCache().clear()
        print("Cache cleared!")
        return

    if args.command == "stop":
        handle_stop(args)
    elif args.command == "status":
        handle_status(args)
    elif args.command == "model":
        asyncio.run(_handle_model(args))
    elif args.command == "keys":
        asyncio.run(_handle_keys(args))
    elif args.command == "export":
        asyncio.run(_handle_export(args))
    else:
        # Default action: start proxy
        handle_start(args)


if __name__ == "__main__":
    main()
