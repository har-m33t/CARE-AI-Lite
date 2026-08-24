"""Blinding: the export must not let a rater work out the condition."""

from __future__ import annotations

import pytest

from carelite.eval.human import (
    Assignment,
    BlindingViolation,
    RateableItem,
    assert_blinded,
    build_packet,
    rater_seed,
    unblind,
)
from carelite.eval.human.blinding import BlindedPacket, BlindItem
from carelite.eval.rubric.calibration import CALIBRATION_SET
from carelite.types import Condition

CONDITIONS = (Condition.A, Condition.B, Condition.C)


def items(n: int = 60) -> list[RateableItem]:
    return [
        RateableItem(
            generation_id=f"gen-{i:04d}",
            scenario_text=f"Patient turn {i % 20}.",
            response_text=f"Clinician response number {i}.",
            condition=CONDITIONS[i % 3],
            scenario_id=f"sc-{i % 20:04d}",
        )
        for i in range(n)
    ]


class TestPacketShape:
    def test_calibration_comes_first_in_its_taught_order(self) -> None:
        packet = build_packet("R01", items())
        cal = packet.calibration_items
        assert len(cal) == len(CALIBRATION_SET)
        assert [c.display_order for c in cal] == [1, 2, 3, 4, 5]
        assert packet.study_items[0].display_order == 6
        cal_assignments = [a for a in packet.assignments if a.is_calibration]
        assert [a.calibration_id for a in cal_assignments] == [c.item_id for c in CALIBRATION_SET]
        # And never in generation_id, which is an FK these fixtures cannot satisfy.
        assert all(a.generation_id is None for a in cal_assignments)

    def test_every_study_item_gets_exactly_one_assignment(self) -> None:
        packet = build_packet("R01", items())
        study = [a for a in packet.assignments if not a.is_calibration]
        assert len(study) == 60
        assert len({a.generation_id for a in study}) == 60
        assert len({a.blind_label for a in packet.assignments}) == 65

    def test_calibration_can_be_omitted_for_a_retest(self) -> None:
        packet = build_packet("R01-t2", items(), include_calibration=False)
        assert packet.calibration_items == ()

    def test_duplicate_generation_ids_are_refused(self) -> None:
        dupes = [*items(2), items(2)[0]]
        with pytest.raises(ValueError, match="duplicate generation_id"):
            build_packet("R01", dupes)


class TestBlinding:
    def test_rater_facing_items_carry_no_identifiers(self) -> None:
        """A `BlindItem` has five fields and none of them is a generation id."""
        packet = build_packet("R01", items())
        for item in packet.items:
            blob = f"{item.blind_label} {item.scenario_text} {item.response_text}"
            assert "gen-" not in blob
            for condition in Condition:
                assert f"condition {condition.value}".casefold() not in blob.casefold()

    def test_order_is_randomised_not_input_order(self) -> None:
        source = items()
        packet = build_packet("R01", source)
        presented = [a.generation_id for a in packet.assignments if not a.is_calibration]
        assert presented != [i.generation_id for i in source]
        assert sorted(presented) == sorted(i.generation_id for i in source)

    def test_two_raters_get_different_orders(self) -> None:
        source = items()
        a = [x.generation_id for x in build_packet("R01", source).assignments]
        b = [x.generation_id for x in build_packet("R02", source).assignments]
        assert a != b

    def test_the_same_rater_gets_a_reproducible_packet(self) -> None:
        """A rater who loses their file gets it back with the same labels."""
        source = items()
        first = build_packet("R01", source)
        second = build_packet("R01", source)
        assert first.label_to_generation() == second.label_to_generation()

    def test_rater_seed_is_stable_across_processes(self) -> None:
        # blake2b, not hash(): PYTHONHASHSEED would otherwise reshuffle every run.
        assert rater_seed("R01") == rater_seed("R01")
        assert rater_seed("R01") != rater_seed("R02")

    def test_a_blocked_presentation_order_is_rejected(self) -> None:
        """The leak that survives code review: every field stripped, order sorted."""
        source = sorted(items(), key=lambda i: str(i.condition))
        blocked = BlindedPacket(
            rater_id="R01",
            seed=1,
            items=tuple(
                BlindItem(f"R01-{i:03d}", i, False, s.scenario_text, s.response_text)
                for i, s in enumerate(source, start=1)
            ),
            assignments=tuple(
                Assignment("R01", s.generation_id, i, f"R01-{i:03d}")
                for i, s in enumerate(source, start=1)
            ),
        )
        with pytest.raises(BlindingViolation, match="blocks"):
            assert_blinded(blocked, source)

    def test_a_leaked_generation_id_is_rejected(self) -> None:
        source = items(4)
        leaky = BlindedPacket(
            rater_id="R01",
            seed=1,
            items=(BlindItem("R01-001", 1, False, "turn", f"see {source[0].generation_id}"),),
            assignments=(Assignment("R01", source[0].generation_id, 1, "R01-001"),),
        )
        with pytest.raises(BlindingViolation, match="generation id"):
            assert_blinded(leaky, source)

    def test_a_correct_packet_passes_its_own_audit(self) -> None:
        source = items()
        assert_blinded(build_packet("R01", source), source)

    def test_a_tiny_packet_is_not_falsely_flagged(self) -> None:
        """Three items from three conditions give three runs however well shuffled."""
        source = items(3)
        assert_blinded(build_packet("R01", source), source)


class TestAssignmentInvariant:
    """The one schema-level invariant of this harness, checked without Postgres.

    `rating_assignment` enforces these three rules with CHECK constraints, but
    those only fire where a database is running. The harness is developed, and
    `make check` is run, mostly where one is not — so the same three rules are
    enforced in `Assignment.__post_init__` and asserted here, in a test that is
    not marked `db` and therefore actually runs. The db-marked round trip in
    `test_store_db.py` proves the constraints agree with these; this proves the
    rules are enforced at all on an ordinary run.
    """

    def test_both_targets_null_is_refused(self) -> None:
        with pytest.raises(ValueError, match="exactly one target"):
            Assignment("R01", None, 1, "R01-001")

    def test_both_targets_set_is_refused(self) -> None:
        with pytest.raises(ValueError, match="exactly one target"):
            Assignment("R01", "gen-0001", 1, "R01-001", True, "CAL-01")

    def test_flag_disagreeing_with_the_target_is_refused(self) -> None:
        with pytest.raises(ValueError, match="flag and the target must agree"):
            Assignment("R01", "gen-0001", 1, "R01-001", is_calibration=True)
        with pytest.raises(ValueError, match="flag and the target must agree"):
            Assignment("R01", None, 1, "R01-C01", is_calibration=False, calibration_id="CAL-01")

    def test_a_valid_calibration_assignment_is_accepted(self) -> None:
        a = Assignment.for_calibration("R01", "CAL-01", 1, "R01-C01")
        assert (a.generation_id, a.calibration_id, a.is_calibration) == (None, "CAL-01", True)
        assert a.target_id == "CAL-01"

    def test_a_valid_study_assignment_is_accepted(self) -> None:
        a = Assignment.for_generation("R01", "gen-0001", 6, "R01-001")
        assert (a.generation_id, a.calibration_id, a.is_calibration) == ("gen-0001", None, False)
        assert a.target_id == "gen-0001"


class TestUnblind:
    def test_join_recovers_generation_ids(self) -> None:
        packet = build_packet("R01", items(6))
        ratings = {a.blind_label: {"de": 4} for a in packet.assignments if not a.is_calibration}
        joined = unblind(packet.assignments, ratings)
        assert sorted(joined) == sorted(
            a.generation_id for a in packet.assignments if not a.is_calibration
        )

    def test_an_unknown_label_is_refused(self) -> None:
        packet = build_packet("R01", items(6))
        with pytest.raises(KeyError, match="unknown blind label"):
            unblind(packet.assignments, {"R02-001": {"de": 4}})

    def test_calibration_labels_are_dropped_not_unblinded(self) -> None:
        """The quiet defect: a calibration id landing in a dict of generation ids.

        Every rater scores the five calibration items against a consensus they
        are then shown, so a `CAL-0x` key here becomes a near-unanimous unit in
        the alpha computation. That raises agreement rather than erroring, which
        is why it is dropped at the join instead of being left to each caller.
        """
        packet = build_packet("R01", items(6))
        ratings = {a.blind_label: {"de": 4} for a in packet.assignments}
        joined = unblind(packet.assignments, ratings)

        cal_ids = {c.item_id for c in CALIBRATION_SET}
        assert not cal_ids & set(joined)
        assert None not in joined
        assert len(joined) == 6

    def test_a_calibration_label_is_recognised_rather_than_unknown(self) -> None:
        """Dropped is not the same as unknown, and the error must keep saying so."""
        packet = build_packet("R01", items(6))
        cal_label = packet.assignments[0].blind_label
        assert unblind(packet.assignments, {cal_label: {"de": 4}}) == {}

    def test_label_to_generation_is_study_only(self) -> None:
        packet = build_packet("R01", items(6))
        assert len(packet.label_to_generation()) == 6
        assert len(packet.label_to_calibration()) == len(CALIBRATION_SET)
        assert not set(packet.label_to_generation()) & set(packet.label_to_calibration())
