"""``aihydro-watershed signatures``: offline CLI over the signatures kernel."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from aihydro_watershed import cli
from aihydro_watershed.signatures.signatures import extract_hydrological_signatures

SQUARE = {
    "type": "Polygon",
    "coordinates": [[[-77.5, 39.2], [-77.4, 39.2], [-77.4, 39.3], [-77.5, 39.3], [-77.5, 39.2]]],
}


def _series(n_years: int = 3) -> pd.Series:
    n = 365 * n_years
    rng = np.random.default_rng(0)
    values = 5.0 + 3.0 * np.sin(np.arange(n) / 58.0) + rng.gamma(2.0, 1.0, n)
    return pd.Series(values, index=pd.date_range("2000-10-01", periods=n, freq="D"), name="q_cms")


@pytest.fixture
def files(tmp_path):
    q = _series()
    csv = tmp_path / "q.csv"
    pd.DataFrame({"date": q.index.strftime("%Y-%m-%d"), "q_cms": q.values}).to_csv(csv, index=False)
    gj = tmp_path / "basin.geojson"
    gj.write_text(json.dumps({"type": "Feature", "properties": {}, "geometry": SQUARE}))
    return q, csv, gj


def test_signatures_matches_python_api_and_makes_no_network_call(files, capsys, monkeypatch):
    q, csv, gj = files

    def refuse(*_a, **_k):
        raise AssertionError("network access attempted by the offline CLI")

    import socket
    monkeypatch.setattr(socket.socket, "connect", refuse)

    rc = cli.main(["signatures", "--flow-csv", str(csv), "--area-km2", "250", "--geojson", str(gj)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)

    direct = extract_hydrological_signatures(
        None, SQUARE, 250.0, start_date=str(q.index.min().date()),
        end_date=str(q.index.max().date()), q_cms_series=q, precipitation="skip",
    )
    for key in ("q_mean", "q5", "q95", "baseflow_index", "slope_fdc"):
        assert out["data"][key] == pytest.approx(direct.data[key], rel=1e-12), key
    # Precipitation is never requested, so the water-balance signatures are null.
    assert out["data"]["runoff_ratio"] is None
    assert out["data"]["_precipitation"]["status"] == "not_attempted"
    assert out["meta"]["tool"].endswith("extract_hydrological_signatures")


def test_output_file_and_custom_columns(files, tmp_path, capsys):
    q, _csv, gj = files
    alt = tmp_path / "alt.csv"
    pd.DataFrame({"day": q.index.strftime("%Y-%m-%d"), "flow": q.values}).to_csv(alt, index=False)
    dest = tmp_path / "out.json"
    rc = cli.main(["signatures", "--flow-csv", str(alt), "--date-col", "day", "--flow-col", "flow",
                   "--area-km2", "250", "--geojson", str(gj), "-o", str(dest)])
    assert rc == 0 and capsys.readouterr().out == ""
    assert json.loads(dest.read_text())["data"]["q_mean"] > 0


def test_missing_column_and_bad_geometry_are_one_line_errors(files, tmp_path, capsys):
    _q, csv, gj = files
    rc = cli.main(["signatures", "--flow-csv", str(csv), "--flow-col", "nope",
                   "--area-km2", "250", "--geojson", str(gj)])
    assert rc == 1 and "no column 'nope'" in capsys.readouterr().err

    point = tmp_path / "pt.geojson"
    point.write_text(json.dumps({"type": "Point", "coordinates": [0, 0]}))
    rc = cli.main(["signatures", "--flow-csv", str(csv), "--area-km2", "250", "--geojson", str(point)])
    assert rc == 1 and "Polygon" in capsys.readouterr().err


def test_nonpositive_area_is_refused(files, capsys):
    _q, csv, gj = files
    rc = cli.main(["signatures", "--flow-csv", str(csv), "--area-km2", "0", "--geojson", str(gj)])
    err = capsys.readouterr().err
    assert rc == 1 and "area" in err.lower()


def test_no_subcommand_prints_help(capsys):
    assert cli.main([]) == 0
    assert "signatures" in capsys.readouterr().out
