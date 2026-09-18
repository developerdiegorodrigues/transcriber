from types import SimpleNamespace
from unittest import TestCase

from transcriber.quality import annotate_quality, quality_reasons, retry_ranges


def segment(start=0.0, end=1.0, logprob=-0.2, compression=1.0, no_speech=0.1, temp=0.0):
    return SimpleNamespace(
        start=start,
        end=end,
        avg_logprob=logprob,
        compression_ratio=compression,
        no_speech_prob=no_speech,
        temperature=temp,
    )


class QualityTests(TestCase):
    def test_detects_each_low_confidence_signal(self):
        reasons = quality_reasons(
            segment(logprob=-2.0, compression=3.0, no_speech=0.8, temp=1.0)
        )
        self.assertEqual(len(reasons), 4)

    def test_groups_adjacent_suspicious_segments(self):
        segments = [
            segment(0, 2),
            segment(2, 4, logprob=-2.0),
            segment(4.2, 6, temp=1.0),
            segment(20, 22, compression=3.0),
        ]

        ranges = retry_ranges(segments, max_ranges=3)

        self.assertEqual(len(ranges), 2)
        self.assertEqual(ranges[0].indices, (1, 2))
        self.assertEqual((ranges[1].start, ranges[1].end), (20.0, 22.0))

    def test_annotation_returns_summary(self):
        segments = [
            {
                "avg_logprob": -0.2,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.1,
                "temperature": 0.0,
            },
            {
                "avg_logprob": -2.0,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.1,
                "temperature": 0.0,
            },
        ]
        summary = annotate_quality(segments)
        self.assertEqual(summary["low_confidence_segments"], 1)
        self.assertEqual(segments[1]["quality"]["status"], "low_confidence")
