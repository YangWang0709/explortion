#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import re
import subprocess
import sys
from pathlib import Path


def clear_proxy_env():
    env = os.environ.copy()
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(key, None)
    return env


def run(cmd, env, capture=False):
    kwargs = {
        "env": env,
        "text": True,
        "stdout": subprocess.PIPE if capture else None,
        "stderr": subprocess.STDOUT if capture else None,
    }
    result = subprocess.run(cmd, **kwargs)
    if result.returncode != 0:
        if capture and result.stdout:
            print(result.stdout, file=sys.stderr)
        raise RuntimeError("command failed: {}".format(" ".join(cmd)))
    return result.stdout if capture else ""


def get_content_length(url, env):
    output = run(
        [
            "curl",
            "-L",
            "-I",
            "-sS",
            "--connect-timeout",
            "20",
            "--max-time",
            "120",
            url,
        ],
        env,
        capture=True,
    )
    lengths = []
    for line in output.splitlines():
        match = re.match(r"content-length:\s*(\d+)", line, flags=re.IGNORECASE)
        if match:
            lengths.append(int(match.group(1)))
    if not lengths:
        raise RuntimeError("could not find content-length in curl headers")
    return lengths[-1]


def download_part(index, start, end, url, parts_dir, env):
    expected = end - start + 1
    part = parts_dir / "part_{:05d}".format(index)
    if part.exists() and part.stat().st_size == expected:
        return index, expected, "skip"

    tmp = parts_dir / "part_{:05d}.tmp".format(index)
    for attempt in range(1, 21):
        have = tmp.stat().st_size if tmp.exists() else 0
        if have == expected:
            break
        if have > expected:
            raise RuntimeError(
                "part {} expected at most {} bytes, got {}".format(
                    index, expected, have
                )
            )

        offset = start + have
        cmd = [
            "curl",
            "-L",
            "--fail",
            "-sS",
            "--connect-timeout",
            "30",
            "--range",
            "{}-{}".format(offset, end),
            url,
        ]
        with tmp.open("ab") as dst:
            result = subprocess.run(
                cmd,
                env=env,
                stdout=dst,
                stderr=subprocess.PIPE,
                text=True,
            )
        if result.returncode == 0:
            continue
        if attempt == 20:
            raise RuntimeError(
                "part {} failed after {} attempts: {}".format(
                    index, attempt, result.stderr.strip()
                )
            )
        if result.stderr.strip():
            print(
                "part {} attempt {} failed: {}".format(
                    index, attempt, result.stderr.strip()
                ),
                flush=True,
            )

    actual = tmp.stat().st_size
    if actual != expected:
        raise RuntimeError(
            "part {} expected {} bytes, got {}".format(index, expected, actual)
        )
    tmp.replace(part)
    return index, expected, "download"


def merge_parts(parts, output):
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()
    with tmp_output.open("wb") as dst:
        for part in parts:
            with part.open("rb") as src:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
    tmp_output.replace(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mb", type=int, default=16)
    args = parser.parse_args()

    url = (
        "https://drive.usercontent.google.com/download"
        "?id={}&export=download&confirm=t".format(args.file_id)
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    parts_dir = Path(str(output) + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    env = clear_proxy_env()
    total = get_content_length(url, env)
    chunk_size = args.chunk_mb * 1024 * 1024
    ranges = []
    for index, start in enumerate(range(0, total, chunk_size)):
        end = min(start + chunk_size - 1, total - 1)
        ranges.append((index, start, end))

    print("url={}".format(url))
    print("output={}".format(output))
    print("total_bytes={}".format(total))
    print("parts={}".format(len(ranges)))
    print("workers={}".format(args.workers))
    print("chunk_mb={}".format(args.chunk_mb))

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_part, index, start, end, url, parts_dir, env)
            for index, start, end in ranges
        ]
        for future in concurrent.futures.as_completed(futures):
            index, expected, status = future.result()
            completed += 1
            done_bytes = sum(
                (parts_dir / "part_{:05d}".format(i)).stat().st_size
                for i, _, _ in ranges
                if (parts_dir / "part_{:05d}".format(i)).exists()
            )
            print(
                "part {}/{} {} bytes {} complete; downloaded_bytes={}/{}".format(
                    index + 1, len(ranges), expected, status, done_bytes, total
                ),
                flush=True,
            )

    part_paths = [parts_dir / "part_{:05d}".format(index) for index, _, _ in ranges]
    missing = [str(path) for path in part_paths if not path.exists()]
    if missing:
        raise RuntimeError("missing parts: {}".format(missing[:5]))
    merge_parts(part_paths, output)
    print("merged={}".format(output))
    print("merged_size={}".format(output.stat().st_size))


if __name__ == "__main__":
    main()
