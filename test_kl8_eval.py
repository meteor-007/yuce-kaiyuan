from kl8_backtest import split_eval_periods
from kl8_core import Dataset, Draw
from kl8_report import run_pipeline


def _mk_ds():
    draws = []
    for idx in range(60):
        start = (idx * 7) % 80 + 1
        nums = frozenset((start + i) % 80 + 1 for i in range(20))
        draws.append(Draw(date=f"2024-01-{idx + 1:02d}", period=1000 + idx, numbers=nums))
    point_sets = {p: frozenset((p % 80) + 1 + i for i in range(20)) for p in range(1000, 1006)}
    return Dataset(
        draws=draws,
        draws_by_period={d.period: d for d in draws},
        point_sets=point_sets,
        point_dates={p: f"2024-01-{i:02d}" for i, p in enumerate(range(1000, 1006), 1)},
        target_period=1059,
    )


def test_no_pseudo_point_in_prediction_window():
    ds = _mk_ds()
    target = 1059
    _, ctx = run_pipeline(ds, point_period=target)
    assert target not in ctx["ds"].point_sets
    assert target not in ctx["ds"].pseudo_point_periods


def test_no_pseudo_point_for_missing_period():
    ds = _mk_ds()
    target = 9999
    _, ctx = run_pipeline(ds, point_period=target)
    assert target not in ctx["ds"].point_sets
    assert target not in ctx["ds"].pseudo_point_periods


def test_strict_split_has_no_overlap():
    ds = _mk_ds()
    split = split_eval_periods(ds)
    assert split["train"] and split["valid"] and split["test"]
    assert set(split["train"]) & set(split["valid"]) == set()
    assert set(split["valid"]) & set(split["test"]) == set()
    assert set(split["train"]) & set(split["test"]) == set()
