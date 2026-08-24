"""Corpus packing for condition LC, and the truncation it has to admit to."""

from __future__ import annotations

import pytest

from carelite.generate.longcontext import CorpusUnits, build_pack, pack_units
from carelite.generate.model import estimate_tokens
from carelite.types import RetrievedItem


def _item(ref: str, kind: str, words: int) -> RetrievedItem:
    return RetrievedItem(ref_id=ref, kind=kind, text=("word " * words).strip(), score=0.0)


def _units(n_papers: int = 3, per_paper: int = 4, kb: int = 5) -> CorpusUnits:
    chunks = tuple(
        tuple(_item(f"p{p}-c{c}", "chunk", 40) for c in range(per_paper)) for p in range(n_papers)
    )
    return CorpusUnits(
        kb=tuple(_item(f"kb-{i}", "kb_entry", 20) for i in range(kb)),
        chunks_by_paper=chunks,
        n_chunks_total=n_papers * per_paper,
        kb_total=kb,
    )


def test_a_generous_budget_includes_everything_and_says_it_is_not_truncated() -> None:
    pack = pack_units(_units(), budget_tokens=1_000_000)
    assert pack.n_kb_included == 5
    assert pack.n_chunks_included == 12
    assert pack.truncated is False
    assert pack.coverage["chunk_fraction"] == 1.0


def test_the_knowledge_base_goes_in_before_any_chunk() -> None:
    pack = pack_units(_units(), budget_tokens=1_000_000)
    kinds = [item.kind for item in pack.items]
    assert kinds[:5] == ["kb_entry"] * 5


def test_chunks_are_taken_round_robin_so_every_paper_is_represented() -> None:
    """Reading straight through in paper order would fill the window with
    whichever papers sort first and drop the rest of the corpus entirely."""
    units = _units(n_papers=3, per_paper=4, kb=0)
    one_chunk = estimate_tokens(units.chunks_by_paper[0][0].text) + 16
    pack = pack_units(units, budget_tokens=3 * one_chunk)
    assert pack.n_chunks_included == 3
    assert {item.ref_id.split("-")[0] for item in pack.items} == {"p0", "p1", "p2"}
    assert {item.ref_id.split("-")[1] for item in pack.items} == {"c0"}


def test_a_tight_budget_truncates_and_reports_it() -> None:
    pack = pack_units(_units(), budget_tokens=200)
    assert pack.truncated is True
    assert pack.n_chunks_included < pack.n_chunks_total
    assert pack.coverage["truncated"] is True
    assert 0.0 <= pack.coverage["chunk_fraction"] < 1.0


def test_the_budget_is_never_exceeded() -> None:
    for budget in (100, 500, 2_000, 10_000):
        pack = pack_units(_units(n_papers=6, per_paper=8, kb=20), budget_tokens=budget)
        assert pack.est_tokens <= budget


def test_packing_is_deterministic() -> None:
    units = _units(n_papers=4, per_paper=5, kb=7)
    first = pack_units(units, budget_tokens=900)
    second = pack_units(units, budget_tokens=900)
    assert [i.ref_id for i in first.items] == [i.ref_id for i in second.items]


@pytest.mark.db
def test_the_real_corpus_does_not_fit_and_the_pack_says_so() -> None:
    """Read-only. The build plan says the corpus fits in a 256K window; against
    the loaded database and the 128K window this project configures, it does
    not, and condition LC has to carry that as a stated limitation."""
    pack = build_pack()
    assert pack.n_kb_included == pack.n_kb_total, "every KB entry must survive the budget"
    assert pack.n_chunks_total > 0
    assert pack.est_tokens <= pack.budget_tokens
