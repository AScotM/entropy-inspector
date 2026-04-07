#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class ByteFrequency:
    byte: int
    count: int


@dataclass(frozen=True)
class EntropyReport:
    path: str
    size: int
    sha256: str
    entropy: float
    printable_ratio: float
    null_ratio: float
    likely_text: bool
    top_bytes: List[ByteFrequency]


class FileEntropyAnalyzer:
    def __init__(self, chunk_size: int = 1024 * 1024) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size

    def analyze(self, path: Path) -> EntropyReport:
        if not path.exists():
            raise FileNotFoundError(str(path))
        if not path.is_file():
            raise ValueError(f"not a regular file: {path}")

        sha256 = hashlib.sha256()
        counts: Counter[int] = Counter()
        total = 0
        printable = 0
        nulls = 0

        with path.open("rb") as f:
            for chunk in self._read_chunks(f):
                sha256.update(chunk)
                counts.update(chunk)
                total += len(chunk)
                printable += sum(1 for b in chunk if 32 <= b <= 126 or b in (9, 10, 13))
                nulls += chunk.count(0)

        entropy = self._shannon_entropy(counts, total)
        printable_ratio = (printable / total) if total else 0.0
        null_ratio = (nulls / total) if total else 0.0
        likely_text = printable_ratio >= 0.85 and null_ratio == 0.0 and entropy <= 7.5

        top_bytes = [
            ByteFrequency(byte=byte, count=count)
            for byte, count in counts.most_common(10)
        ]

        return EntropyReport(
            path=str(path),
            size=total,
            sha256=sha256.hexdigest(),
            entropy=entropy,
            printable_ratio=printable_ratio,
            null_ratio=null_ratio,
            likely_text=likely_text,
            top_bytes=top_bytes,
        )

    def _read_chunks(self, fh) -> Iterable[bytes]:
        while True:
            chunk = fh.read(self.chunk_size)
            if not chunk:
                break
            yield chunk

    @staticmethod
    def _shannon_entropy(counts: Counter[int], total: int) -> float:
        if total <= 0:
            return 0.0

        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log2(p)
        return entropy


def human_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    idx = 0
    while size >= 1024.0 and idx < len(units) - 1:
        size /= 1024.0
        idx += 1
    return f"{size:.2f} {units[idx]}"


def print_report(report: EntropyReport) -> None:
    print(f"path            : {report.path}")
    print(f"size            : {report.size} ({human_bytes(report.size)})")
    print(f"sha256          : {report.sha256}")
    print(f"entropy         : {report.entropy:.4f}")
    print(f"printable_ratio : {report.printable_ratio:.4f}")
    print(f"null_ratio      : {report.null_ratio:.4f}")
    print(f"likely_text     : {'yes' if report.likely_text else 'no'}")
    print("top_bytes       :")
    for item in report.top_bytes:
        print(f"  0x{item.byte:02x}  {item.count}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="entropy-inspector",
        description="Analyze file entropy and byte distribution",
    )
    parser.add_argument("path", help="target file")
    parser.add_argument("--json", action="store_true", help="output JSON")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024 * 1024,
        help="read chunk size in bytes",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        analyzer = FileEntropyAnalyzer(chunk_size=args.chunk_size)
        report = analyzer.analyze(Path(args.path))

        if args.json:
            print(json.dumps(asdict(report), indent=2, ensure_ascii=False))
        else:
            print_report(report)

        return 0

    except FileNotFoundError as exc:
        print(f"file not found: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"io error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
