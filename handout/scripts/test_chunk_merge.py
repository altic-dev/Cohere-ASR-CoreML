#!/usr/bin/env python3
"""Regression tests for long-audio window planning and transcript merge (mirrors ChunkMergeCore)."""
from __future__ import annotations

import unittest


def window_starts(total_samples: int, window_samples: int, overlap_samples: int) -> list[int]:
    if total_samples <= 0 or window_samples <= 0:
        return []
    if overlap_samples < 0 or overlap_samples >= window_samples:
        return []
    if total_samples <= window_samples:
        return [0]
    stride = window_samples - overlap_samples
    starts: list[int] = []
    s = 0
    while s + window_samples < total_samples:
        starts.append(s)
        s += stride
    last_start = total_samples - window_samples
    if not starts or starts[-1] < last_start:
        starts.append(last_start)
    return starts


def merge_two(a: str, b: str) -> str:
    wa = a.split()
    wb = b.split()
    if not wa:
        return b
    if not wb:
        return a
    max_j = min(len(wa), len(wb), 48)
    for j in range(max_j, 0, -1):
        if wa[-j:] == wb[:j]:
            rest = wb[j:]
            if not rest:
                return a
            return a + " " + " ".join(rest)
    return a + " " + b


def merge_transcript_chunks(parts: list[str]) -> str:
    if not parts:
        return ""
    acc = parts[0]
    for nxt in parts[1:]:
        acc = merge_two(acc, nxt)
    return acc.strip()


class TestWindowStarts(unittest.TestCase):
    def test_overlapping_100_40_10(self):
        self.assertEqual(window_starts(100, 40, 10), [0, 30, 60])

    def test_tail_95(self):
        self.assertEqual(window_starts(95, 40, 10), [0, 30, 55])

    def test_short_audio(self):
        self.assertEqual(window_starts(100, 480_000, 80_000), [0])


class TestMerge(unittest.TestCase):
    def test_duplicate_prefix(self):
        self.assertEqual(
            merge_transcript_chunks(["hello world", "world how are you"]),
            "hello world how are you",
        )

    def test_boundary_phrase(self):
        self.assertEqual(
            merge_transcript_chunks(["the quick brown", "quick brown fox"]),
            "the quick brown fox",
        )

    def test_no_overlap(self):
        self.assertEqual(merge_transcript_chunks(["alpha", "beta"]), "alpha beta")


if __name__ == "__main__":
    unittest.main()
