"""数据与命令行测试。"""
from __future__ import annotations

import pytest

from kairos_rates.cli import main
from kairos_rates.curve import bootstrap
from kairos_rates.data import (
    curve_table,
    load_quotes_csv,
    save_quotes_csv,
    synthetic_curve,
    synthetic_quotes,
)


def test_synthetic_quote_roundtrip():
    curve = synthetic_curve()
    quotes = synthetic_quotes(curve)
    boot = bootstrap(quotes)
    err = max(abs(boot.par_rate(q.maturity, q.frequency) - q.rate)
              for q in quotes if q.kind == "swap")
    assert err < 1e-10
    deposit = next(q for q in quotes if q.kind == "deposit")
    assert float(boot.df(deposit.maturity)) == pytest.approx(
        1.0 / (1.0 + deposit.rate * deposit.maturity), rel=1e-12)


def test_csv_roundtrip(tmp_path):
    quotes = synthetic_quotes(synthetic_curve())
    path = tmp_path / "curve.csv"
    save_quotes_csv(quotes, str(path))
    loaded = load_quotes_csv(str(path))
    assert len(loaded) == len(quotes)
    for a, b in zip(quotes, loaded):
        assert a.kind == b.kind
        assert a.maturity == pytest.approx(b.maturity)
        assert a.rate == pytest.approx(b.rate)


def test_curve_table():
    curve = synthetic_curve()
    rows = curve_table(curve)
    assert rows and all("zero" in r and "df" in r for r in rows)


def test_cli_demo(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "bootstrap" in out
    assert "Nelson-Siegel" in out
    assert "关键利率久期" in out


def test_cli_bond_and_swap(capsys):
    assert main(["bond", "--coupon", "0.03", "--maturity", "10"]) == 0
    out = capsys.readouterr().out
    assert "价格" in out and "久期" in out

    assert main(["swap", "--maturity", "5"]) == 0
    out = capsys.readouterr().out
    assert "平价利率" in out and "DV01" in out


def test_cli_curve(tmp_path, capsys):
    path = tmp_path / "curve.csv"
    save_quotes_csv(synthetic_quotes(synthetic_curve()), str(path))
    assert main(["curve", "--csv", str(path)]) == 0
    assert "零息利率" in capsys.readouterr().out


def test_cli_version_and_top_level_help(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("kairos-rates ")

    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "{demo,curve,bond,swap}" in out