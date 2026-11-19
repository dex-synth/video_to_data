"""Run the native FlashCHORD controller against an h2r_bench evaluation socket (h2r-evaluation-v2)."""

import argparse
import json
import os
from pathlib import Path
import socket
import time

from h2r_io.client import receive, send

from .controller import Controller
from .inputs import write_json


def expect(stream, kind):
    message = receive(stream)
    if message.get("type") != kind:
        raise ValueError(f"Expected {kind} from the evaluation server, got {message}")
    return message


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
        hello = expect(stream, "hello")
        controller = Controller(args.checkpoint, args.prepared, hello, device=args.device)
        start = controller.start()
        write_json(args.output / "controller.json", {
            "binding": start["binding"], "prepared": str(args.prepared.resolve()), "task_id": hello["task_id"],
            "control_dt_s": controller.control_dt, "control_ticks": controller.control_ticks,
            "deterministic": True, "device": args.device, "policy_batch_size": 1,
            "xla_flags": os.environ["XLA_FLAGS"], "hello": hello, "checkpoint_config": controller.saved_config})
        send(stream, start)
        with (args.output / "controls.jsonl").open("x") as log:
            while True:
                message = expect(stream, "frames")
                if message["done"]:
                    write_json(args.output / "result.json", message)
                    break
                begin = time.monotonic()
                command, diagnostics = controller.step(message["episodes"])
                log.write(json.dumps({"command": command, "compute_s": time.monotonic() - begin,
                                      **diagnostics}, allow_nan=False) + "\n")
                log.flush()
                send(stream, command)
    for episode in message["episodes"]:
        print(f"episode {episode['episode_id']} (seed {episode['seed']}): {episode.get('reason')} "
              f"at tick {episode['tick']}", flush=True)


if __name__ == "__main__":
    main()
