"""Tests for pdf_pipeline utility functions."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdf_pipeline import parse_slide_range


class TestParseSlideRange:
    """Tests for parse_slide_range() — converts user slide specs to 0-based indices."""

    def test_single_slide(self):
        """A single slide number returns a one-element list."""
        assert parse_slide_range("3", 5) == [2]

    def test_single_slide_first(self):
        """Slide 1 maps to index 0."""
        assert parse_slide_range("1", 5) == [0]

    def test_single_slide_last(self):
        """Last slide maps to total-1."""
        assert parse_slide_range("5", 5) == [4]

    def test_range(self):
        """A dash range returns all indices in that range, inclusive."""
        assert parse_slide_range("1-3", 5) == [0, 1, 2]

    def test_range_full(self):
        """A range spanning all slides returns every index."""
        assert parse_slide_range("1-5", 5) == [0, 1, 2, 3, 4]

    def test_range_single_element(self):
        """A range of one slide (e.g. '3-3') returns one index."""
        assert parse_slide_range("3-3", 5) == [2]

    def test_comma_list(self):
        """Comma-separated slides return each mapped index, sorted."""
        assert parse_slide_range("1,3,5", 5) == [0, 2, 4]

    def test_mixed_range_and_list(self):
        """Mixed range and comma syntax are combined correctly."""
        assert parse_slide_range("1-3,5", 5) == [0, 1, 2, 4]

    def test_result_is_sorted(self):
        """Output is always sorted ascending regardless of input order."""
        assert parse_slide_range("5,1,3", 5) == [0, 2, 4]

    def test_duplicates_deduplicated(self):
        """Overlapping range and explicit slide produce no duplicate indices."""
        assert parse_slide_range("1-3,2", 5) == [0, 1, 2]

    def test_range_clamped_at_total(self):
        """Range end exceeding total is clamped to last slide."""
        assert parse_slide_range("3-99", 5) == [2, 3, 4]

    def test_range_clamped_at_start(self):
        """Range start of 0 is clamped to 1 (slides are 1-based)."""
        assert parse_slide_range("0-3", 5) == [0, 1, 2]

    def test_out_of_range_single_slide(self):
        """A slide number beyond total is silently excluded."""
        assert parse_slide_range("10", 5) == []

    def test_inverted_range(self):
        """A range where start > end after clamping produces an empty list."""
        assert parse_slide_range("5-2", 5) == []

    def test_whitespace_around_parts(self):
        """Spaces around commas and numbers are stripped."""
        assert parse_slide_range(" 1-3 , 5 ", 5) == [0, 1, 2, 4]

    def test_single_slide_total_one(self):
        """Works correctly for a one-page PDF."""
        assert parse_slide_range("1", 1) == [0]
