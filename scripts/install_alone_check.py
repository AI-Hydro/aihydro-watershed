"""Install-alone task for aihydro-watershed.

Run by ``.github/workflows/install-alone.yml`` with the *venv's* interpreter,
from a directory outside the checkout, after installing ONLY the built wheel
(no extras) and its PyPI dependencies (aihydro-core, aihydro-data among them).
No other AI-Hydro package is present and every socket connection is refused.

Task: hydrological signatures from a local streamflow file through the
``aihydro-watershed signatures`` CLI.
  1. the console script is installed and reports the wheel's version;
  2. a 12-water-year synthetic daily record + a basin polygon (written to a
     temp dir) go through the CLI in-process, network refused;
  3. the JSON carries the signature set, a bootstrap CI (>= 10 complete water
     years), and records that precipitation was never requested.

The series is synthetic and only exercises the install; no value printed here
is a hydrological result.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile


def _block_network() -> None:
    def refuse(*_a, **_k):
        raise RuntimeError("network access attempted during an install-alone check")

    socket.socket.connect = refuse  # type: ignore[assignment]
    socket.create_connection = refuse  # type: ignore[assignment]
    socket.getaddrinfo = refuse  # type: ignore[assignment]


def main() -> int:
    import aihydro_watershed

    where = aihydro_watershed.__file__
    assert "site-packages" in where, f"imported from a source tree, not the wheel: {where}"

    exe = os.path.join(os.path.dirname(sys.executable), "aihydro-watershed")
    ver = subprocess.run([exe, "--version"], check=True, capture_output=True, text=True).stdout
    assert aihydro_watershed.__version__ in ver, ver

    _block_network()
    import numpy as np
    import pandas as pd

    from aihydro_watershed.cli import main as cli_main

    n = 365 * 12 + 3  # 2000-10-01 .. 2012-10-03: 12 complete water years
    rng = np.random.default_rng(42)
    q = 5.0 + 3.0 * np.sin(np.arange(n) / 58.0) + rng.gamma(2.0, 1.0, n)
    days = pd.date_range("2000-10-01", periods=n, freq="D")
    square = {"type": "Polygon", "coordinates": [[
        [-77.5, 39.2], [-77.4, 39.2], [-77.4, 39.3], [-77.5, 39.3], [-77.5, 39.2]]]}

    with tempfile.TemporaryDirectory() as tmp:
        csv, gj, out = (os.path.join(tmp, f) for f in ("q.csv", "basin.geojson", "out.json"))
        pd.DataFrame({"date": days.strftime("%Y-%m-%d"), "q_cms": q}).to_csv(csv, index=False)
        with open(gj, "w") as fh:
            json.dump({"type": "Feature", "properties": {}, "geometry": square}, fh)
        rc = cli_main(["signatures", "--flow-csv", csv, "--area-km2", "250",
                       "--geojson", gj, "-o", out])
        assert rc == 0, f"CLI exited {rc}"
        with open(out) as fh:
            res = json.load(fh)

    data = res["data"]
    for key in ("q_mean", "q5", "q95", "baseflow_index", "slope_fdc", "hfd_mean"):
        assert isinstance(data[key], float), (key, data.get(key))
    assert data["q5"] < data["q95"]
    assert data["runoff_ratio"] is None
    assert data["_precipitation"]["status"] == "not_attempted", data["_precipitation"]
    ci = data["_uncertainty"]["q_mean"]
    # q_mean in `data` is the series mean in mm/day; the CI is of the same statistic.
    assert not ci.get("ci_refused") and ci["ci_low"] < ci["value"] < ci["ci_high"], ci
    assert ci["n_groups"] == 12, ci

    print(f"aihydro-watershed {aihydro_watershed.__version__}: install-alone task OK ({where})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
