"""Run the native FlashCHORD controller against an h2r_bench Unix socket."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import struct
import time

from .controller import Controller

MAX_MESSAGE_BYTES = 32 * 1024 * 1024


def send(stream, value):
    payload = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Protocol message too large")
    stream.sendall(struct.pack("!I", len(payload)) + payload)


def receive(stream):
    def read(count):
        result = bytearray()
        while len(result) < count:
            block = stream.recv(count - len(result))
            if not block:
                raise EOFError("Evaluation server disconnected")
            result.extend(block)
        return result

    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"Duplicate field: {key}")
            result[key] = value
        return result

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Non-finite protocol value")
        return result

    size, = struct.unpack("!I", read(4))
    if not 0 < size <= MAX_MESSAGE_BYTES:
        raise ValueError("Invalid protocol message size")
    value = json.loads(read(size), object_pairs_hook=pairs, parse_constant=number, parse_float=number)
    if not isinstance(value, dict) or value.get("type") == "error":
        raise ValueError(f"Invalid server response: {value}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="CUDA device for native observations, actions and actor")
    parser.add_argument("--timeout-s", type=float, default=120)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.settimeout(args.timeout_s)
        stream.connect(args.socket)
        hello = receive(stream)
        if hello.get("type") != "hello":
            raise ValueError("Expected hello")
        controller = Controller(args.checkpoint, args.prepared, hello, device=args.device)
        with args.checkpoint.open("rb") as checkpoint:
            digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
        manifest = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": digest,
                    "prepared": str(args.prepared.resolve()), "task_id": hello["task_id"],
                    "deterministic": True, "device": args.device,
                    "policy_batch_size": 1, "xla_flags": os.environ["XLA_FLAGS"], "hello": hello,
                    "checkpoint_config": controller.saved_config}
        (args.output / "controller.json").write_text(json.dumps(manifest, indent=2) + "\n")
        send(stream, controller.start())
        with (args.output / "controls.jsonl").open("x") as log:
            while True:
                message = receive(stream)
                if message.get("type") != "frames":
                    raise ValueError("Expected frames")
                if message["done"]:
                    (args.output / "result.json").write_text(json.dumps(message, indent=2) + "\n")
                    break
                begin = time.monotonic()
                command, diagnostics = controller.step(message["episodes"])
                log.write(json.dumps({"command": command, "compute_s": time.monotonic() - begin,
                                      **diagnostics}, allow_nan=False) + "\n")
                log.flush()
                send(stream, command)


if __name__ == "__main__":
    main()
