import numpy as np
import pytest

from flynum.experiment import canonical_split
from flynum.graph import Graph, fixture, rewire
from flynum.model import Reservoir, RetinaEncoder, fit_readout, metrics
from flynum.stimuli import check_leakage, make_dot_image, pairs, singles
from flynum.prepare import coalesce_edges, transmitter_polarity


def test_directed_rewire_preserves_degrees_signs_and_source_weights():
    g = fixture(17, n=100)
    r, audit = rewire(g, 42, 3, .5)
    for p, q in ((g.pre, r.pre), (g.post, r.post)):
        np.testing.assert_array_equal(np.bincount(p, minlength=g.n), np.bincount(q, minlength=g.n))
    for sign in [-1, 1]:
        np.testing.assert_array_equal(np.bincount(g.post[g.signs[g.pre] == sign], minlength=g.n),
                                      np.bincount(r.post[r.signs[r.pre] == sign], minlength=g.n))
    np.testing.assert_array_equal(g.pre, r.pre)
    np.testing.assert_array_equal(g.count, r.count)
    assert np.all(r.pre != r.post)
    assert len(set(zip(r.pre, r.post))) == len(r.pre)
    assert audit["changed_edge_fraction"] >= .5


def test_synaptic_direction_and_between_stimulus_memory():
    g = Graph(np.arange(2), np.array([0]), np.array([1]), np.array([1.]), np.array([1, 1]),
              np.array([0]), np.array([[.5, .5]]), np.array(["input", "output"]), {"kind": "test"})
    m = Reservoir(g, {"leak": .5, "gain": .9, "input_scale": 1}, 8)
    on, off = np.ones((1, 1), np.float32), np.zeros((1, 1), np.float32)
    assert m.states(on, duration=1, post_steps=0)[0, 0] == 0
    assert m.states(on, duration=2, post_steps=0)[0, 0] > 0
    remembered = m.states(on, off, gap=1, duration=2, post_steps=0)
    no_history = m.states(off, off, gap=1, duration=2, post_steps=0)
    assert remembered[0, 0] > no_history[0, 0]
    np.testing.assert_array_equal(remembered, m.states(on, off, gap=1, duration=2, post_steps=0))


def test_area_and_conflict_controls_are_measured_not_label_features():
    rng = np.random.default_rng(5)
    fixed = [make_dot_image(n, "fixed_area", rng)[1] for n in range(1, 11)]
    np.testing.assert_allclose([r["geometric_area"] for r in fixed], .048)
    conflict = [make_dot_image(n, "cue_conflict", rng)[1] for n in range(1, 11)]
    assert np.all(np.diff([r["pixel_mass"] for r in conflict]) < 0)
    for n in range(1, 11):
        _, meta = make_dot_image(n, "fixed_radius", rng)
        c = np.array(meta["centers"])
        distances = np.sqrt(((c[:, None] - c[None]) ** 2).sum(2))
        distances[np.diag_indices(n)] = np.inf
        assert distances.min() > 2 * meta["radius"]


def test_data_leakage_audit_exempts_only_blank_images():
    a = singles(1, ["fixed_area"], 2, 24)
    b = singles(2, ["fixed_area"], 2, 24)
    assert check_leakage(a, b)["nonzero_cross_split_duplicates"] == 0
    with pytest.raises(ValueError, match="shared across splits"):
        check_leakage(a, a)


def test_order_swap_uses_same_images_and_group_split_keeps_commutativity():
    p = pairs(7, ["fixed_radius"], 1, 24)
    # Diagonal operands have independent images; compare only off-diagonal order reversals.
    off = p.labels_a != p.labels_b
    np.testing.assert_array_equal(p.a[off], p.b[p.reverse_index[off]])
    train, test = canonical_split(p.groups)
    assert not (set(p.groups[train]) & set(p.groups[test]))
    np.testing.assert_array_equal(train, train[p.reverse_index])


def test_readout_freeze_and_no_target_in_encoder():
    g = fixture(8, 60)
    encoder = RetinaEncoder(g, 24)
    a = singles(19, ["fixed_radius"], 1, 24)
    encoded = encoder(a.images)
    states = Reservoir(g, {"leak": .5, "gain": .9, "input_scale": 1}).states(encoded)
    model = fit_readout(states, a.labels, 1)
    frozen = model.fingerprint()
    model.predict(states * .5)
    assert model.fingerprint() == frozen
    assert encoded.shape == (len(a.images), len(g.input_nodes))
    np.testing.assert_array_equal(encoded, encoder(a.images.copy()))


def test_primary_mae_uses_unclipped_raw_predictions():
    m = metrics(np.array([0, 10]), np.array([-2., 12.]))
    assert m["mae"] == 2
    assert m["exact_accuracy"] == 0


def test_cotransmitter_annotations_and_retinal_identity_are_handled():
    row = {"known_nt": "gaba, nitric oxide", "top_nt": "acetylcholine", "top_nt_conf": ".9"}
    assert transmitter_polarity(row, "Mi4")[0] == -1
    assert transmitter_polarity(row, "R1-6")[1] == "histamine_by_photoreceptor_identity"
    row["known_nt"] = "glutamate, gaba"
    assert transmitter_polarity(row, "unknown")[2] is True


def test_parallel_rows_are_summed_before_synapse_threshold():
    # FlyWire root IDs exceed exact float64 integer precision: keep uint64 throughout.
    a, b, c = 720575940631673939, 720575940619674606, 720575940618728705
    rows = np.array([[a, b, 3], [a, c, 8], [a, b, 4]], dtype=np.uint64)
    merged = coalesce_edges(rows)
    assert len(merged) == 2
    assert merged[(merged[:, 0] == a) & (merged[:, 1] == b), 2][0] == 7
    assert merged.dtype == np.uint64
    assert len(merged[merged[:, 2] >= 5]) == 2


def test_graph_refuses_parallel_rows_instead_of_silently_rewiring_them():
    with pytest.raises(ValueError, match="Parallel edge"):
        Graph(np.arange(2), np.array([0, 0]), np.array([1, 1]), np.ones(2), np.ones(2),
              np.array([0]), np.array([[.5, .5]]), np.array(["a", "b"]), {})
