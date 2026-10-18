"""Tests for the sensor/post-processing layer."""

import csv
import pathlib

import numpy as np
import pytest

from hyperbolic1d.postprocess import (FT, FiberSensor, SensorArray,
                                      export_csv, lsu_fiber_sensors)
from hyperbolic1d.thermal import FormationCoupledWall, lsu_co2_well_stack
from hyperbolic1d.wellbore import ChokeValve, WellboreModel, WellGeometry


@pytest.fixture(scope="module")
def conjugate_run():
    r1, *_ = lsu_co2_well_stack()
    m = WellboreModel(geometry=WellGeometry(diameter=2 * r1),
                      choke=ChokeValve(), gravity=True, friction=True,
                      n_cells=60, wall_model=FormationCoupledWall())
    sensors = lsu_fiber_sensors((100, 1000, 1900))
    res = m.solve(1.379e7, 300.0, 4.0, n_snapshots=3, hist_stride=20,
                  sensors=sensors)
    return m, res


def test_sensor_histories_shapes_and_alignment(conjugate_run):
    _, r = conjugate_run
    n = r.sensor_t.size
    assert n == r.t_hist.size
    for name, v in r.sensor_data.items():
        assert v.size == n, name
    # fluid extras present for fluid sensors only
    assert set(r.sensor_fluid_extra) == {
        n for n in r.sensor_data if r.sensor_meta[n]["location"] == "fluid"}


def test_fluid_sensor_matches_cell_value(conjugate_run):
    m, r = conjugate_run
    # the deepest fluid sensor sits in a specific cell; its last sample
    # must match the fluid temperature of that cell in the final snapshot
    # up to the one-flow-step offset between sampling (pre-update fields)
    # and the snapshot (post-update fields)
    meta = r.sensor_meta["fluid_1900ft"]
    j = int(np.argmin(np.abs(r.z - meta["z"])))
    assert r.sensor_data["fluid_1900ft"][-1] == pytest.approx(
        r.T[-1][j], abs=0.1)


def test_solid_sensor_is_casing_node(conjugate_run):
    m, r = conjugate_run
    meta = r.sensor_meta["fiber13_1000ft"]
    # external fiber mapped near the 13-3/8" casing radius
    assert 0.15 < meta["r"] < 0.175
    # and equals the wall-model node temperature at the end
    wall = m.wall_model
    j = int(np.argmin(np.abs(r.z - meta["z"])))
    k = int(np.argmin(np.abs(wall.radial_positions - meta["r"])))
    assert r.sensor_data["fiber13_1000ft"][-1] == pytest.approx(
        wall.radial_temperatures[j, k], rel=1e-9)


def test_solid_sensor_skipped_without_conjugate_wall():
    m = WellboreModel(n_cells=40)
    sensors = SensorArray([FiberSensor("f", 100.0, "fluid"),
                           FiberSensor("c", 100.0, "casing_ext")])
    with pytest.warns(UserWarning):
        r = m.solve(1.379e7, 300.0, 1.0, n_snapshots=2, sensors=sensors)
    assert "f" in r.sensor_data and "c" not in r.sensor_data


def test_csv_export_roundtrip(conjugate_run, tmp_path):
    _, r = conjugate_run
    out = export_csv(r, tmp_path, profile_times=(0.0, 4.0))
    files = {p.name for p in pathlib.Path(out).iterdir()}
    assert "wellhead_history.csv" in files
    assert "sensors_T.csv" in files
    assert "sensors_fluid_pru.csv" in files
    assert sum(f.startswith("profiles_") for f in files) == 2
    # wellhead history numerically consistent with the result object
    with open(out / "wellhead_history.csv") as f:
        rows = list(csv.reader(f))
    assert rows[0][0] == "t_s"
    data = np.array([[float(x) for x in row[:6]] for row in rows[1:]])
    assert np.allclose(data[:, 0], r.t_hist)
    assert np.allclose(data[:, 4], r.mdot)
    # sensors_T data rows parse and match
    with open(out / "sensors_T.csv") as f:
        rows = [row for row in csv.reader(f) if not row[0].startswith("#")]
    names = rows[0][1:]
    vals = np.array([[float(x) for x in row] for row in rows[1:]])
    for i, n in enumerate(names):
        assert np.allclose(vals[:, i + 1], r.sensor_data[n], rtol=1e-6)


def test_wellhead_face_velocity_recorded(conjugate_run):
    _, r = conjugate_run
    assert r.u_wellhead_face.size == r.t_hist.size
    # face velocity consistent with mdot: u_f = mdot / (rho_f A); since
    # rho_f is not stored, check ordering and positivity instead
    assert np.all(r.u_wellhead_face >= 0.0)
    assert r.u_wellhead_face.max() > r.u_wellhead_face.min()
