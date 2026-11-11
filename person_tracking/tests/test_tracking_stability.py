import pytest

from person_tracking.stability import BoxSmoother, DisplayConfidence, TrackConfirmation, keep_largest_nested_boxes


def test_nested_boxes_keep_only_the_largest_box():
    boxes = [(10, 10, 110, 210), (35, 40, 85, 180), (45, 60, 65, 120)]

    assert keep_largest_nested_boxes(boxes) == [True, False, False]


def test_partial_overlap_between_two_people_is_preserved():
    boxes = [(10, 10, 110, 210), (85, 30, 145, 190)]

    assert keep_largest_nested_boxes(boxes) == [True, True]


@pytest.mark.parametrize(
    "inner_width,expected",
    [(65, [True, False]), (66, [True, True])],
)
def test_nested_box_area_ratio_boundary(inner_width, expected):
    boxes = [(0, 0, 100, 100), (0, 0, inner_width, 100)]

    assert keep_largest_nested_boxes(boxes) == expected


def test_degenerate_box_does_not_suppress_or_get_suppressed():
    boxes = [(0, 0, 100, 100), (20, 20, 20, 80)]

    assert keep_largest_nested_boxes(boxes) == [True, True]


def test_track_confirmation_requires_consecutive_hits_then_survives_short_gaps():
    gate = TrackConfirmation(hits=3, max_gap_seconds=0.1)

    assert gate.update(7, 0.00) == (False, 1)
    assert gate.update(7, 0.04) == (False, 2)
    assert gate.update(7, 0.08) == (True, 3)
    assert gate.update(7, 0.24) == (True, 3)
    gate.forget(7)
    assert gate.update(7, 0.28) == (False, 1)


def test_confirmation_state_can_be_forgotten_when_a_track_expires():
    gate = TrackConfirmation(hits=2, max_gap_seconds=0.1)
    gate.update(4, 1.0)
    gate.forget(4)

    assert gate.update(4, 1.04) == (False, 1)


def test_prune_removes_stale_unconfirmed_tracks_but_keeps_confirmed_tracks():
    gate = TrackConfirmation(hits=2, max_gap_seconds=0.1)
    gate.update(1, 0.0)
    gate.update(2, 0.0)
    gate.update(2, 0.04)

    gate.prune(0.2)

    assert 1 not in gate._states
    assert gate.update(2, 0.24) == (True, 2)


def test_box_smoother_only_changes_smoothed_output():
    smoother = BoxSmoother(alpha=0.5)
    assert smoother.update(1, (0, 0, 10, 20)) == (0, 0, 10, 20)
    assert smoother.update(1, (2, 0, 12, 20)) == pytest.approx((1, 0, 11, 20))


def test_box_smoother_resets_after_a_large_jump_or_track_expiry():
    smoother = BoxSmoother(alpha=0.5)
    smoother.update(1, (0, 0, 10, 10))
    assert smoother.update(1, (100, 100, 110, 110)) == (100, 100, 110, 110)
    smoother.forget(1)
    assert smoother.update(1, (0, 0, 10, 10)) == (0, 0, 10, 10)


def test_display_confidence_survives_small_score_dips_and_requires_reentry():
    gate = DisplayConfidence(0.20, 0.15)
    scores = [0.19, 0.21, 0.19, 0.16, 0.14, 0.18, 0.21]
    assert [gate.update(1, score) for score in scores] == [False, True, True, True, False, False, True]
    assert gate.update(2, 0.18) is False
    gate.forget(1)
    assert gate.update(1, 0.18) is False
    gate.update(1, 0.21)
    gate.reset()
    assert gate.update(1, 0.18) is False


@pytest.mark.parametrize("start,keep", [(0.2, 0.3), (float('nan'), 0.15), (0.2, -0.1)])
def test_display_confidence_rejects_invalid_thresholds(start, keep):
    with pytest.raises(ValueError, match="thresholds"):
        DisplayConfidence(start, keep)
