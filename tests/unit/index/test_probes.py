"""Tests for carelite.index.probes.

Structural checks on `PROBES` (uniqueness, non-empty fields) are pure and
part of `make check`. Actually running the probes needs a live corpus and,
for the dense half, a live embedder -- `@pytest.mark.db` and
`@pytest.mark.inference`. This is the wave-2 retrieval-quality gate.
"""

from __future__ import annotations

import pytest

from carelite.index.probes import PROBES, run_all_probes


def test_there_are_exactly_ten_probes():
    assert len(PROBES) == 10


def test_probe_ids_are_unique():
    ids = [p.probe_id for p in PROBES]
    assert len(ids) == len(set(ids))


def test_every_probe_has_a_nonempty_query_and_expectation():
    for p in PROBES:
        assert p.query.strip()
        assert p.must_contain
        assert all(term.strip() for term in p.must_contain)
        assert p.mode in ("lexical", "dense")
        assert p.target in ("chunk", "kb_entry")
        assert p.top_k > 0


def test_probes_cover_both_lexical_and_dense_modes():
    modes = {p.mode for p in PROBES}
    assert modes == {"lexical", "dense"}


@pytest.mark.db
@pytest.mark.inference
def test_all_ten_probes_pass_against_the_live_corpus():
    results = run_all_probes()
    failed = [r for r in results if not r.passed]
    assert not failed, "\n".join(str(r) for r in failed)
