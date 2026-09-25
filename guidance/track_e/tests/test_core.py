"""Unit tests for the two places a silent bug would invalidate Track E: the delta target and the scramble control.
Run: PYTHONPATH=. python guidance/track_e/tests/test_core.py"""
import numpy as np

from guidance.track_e.core import delta_target, pk_vina, reconstruct, scramble_permutation


def test_pk_vina_known_values():
    assert np.isclose(pk_vina(-13.64), 10.0)          # -13.64 kcal/mol == pK 10
    assert np.isclose(pk_vina(-6.82), 5.0)
    assert pk_vina(0.0) == 0.0


def test_positive_vina_is_clipped_to_zero():
    assert pk_vina(24.86) == 0.0
    assert pk_vina(0.001) == 0.0


def test_delta_definition_and_sign():
    # experimental 8.0, Vina -6.82 (pK 5.0): Vina underestimates -> Delta = +3
    assert np.isclose(delta_target(8.0, -6.82), 3.0)
    # Vina overestimates -> negative delta
    assert np.isclose(delta_target(4.0, -13.64), -6.0)
    # clipped clash pose: delta equals pk itself
    assert np.isclose(delta_target(6.5, 12.0), 6.5)


def test_round_trip_reconstruction_is_exact():
    rng = np.random.RandomState(0)
    pk = rng.uniform(2, 12, 500)
    vina = rng.uniform(-16, 5, 500)
    assert np.allclose(reconstruct(vina, delta_target(pk, vina)), pk, atol=1e-12)


def test_delta_vectorised_matches_scalar():
    pk = np.array([5.0, 7.0, 9.0]); vina = np.array([-5.0, -10.0, 3.0])
    assert np.allclose(delta_target(pk, vina), [delta_target(a, b) for a, b in zip(pk, vina)])


def test_standardisation_round_trip_in_training_units():
    pk = np.array([4.0, 6.0, 9.0, 7.5]); vina = np.array([-6.0, -8.0, -9.0, -3.0])
    d = delta_target(pk, vina); mu, sd = d.mean(), d.std()
    z = (d - mu) / sd
    assert np.allclose(reconstruct(vina, z * sd + mu), pk)


def test_scramble_preserves_size_groups_and_label_multiset():
    rng = np.random.RandomState(1)
    n_lig = rng.randint(8, 14, 400)
    labels = [rng.randint(0, 2, (n, 3)).astype(np.uint8) for n in n_lig]
    src = scramble_permutation(n_lig, seed=7)
    new = [labels[s] for s in src]
    assert all(a.shape == b.shape for a, b in zip(labels, new))            # shapes preserved
    assert np.array_equal(n_lig[src], n_lig)                               # only same-size donors
    for n in np.unique(n_lig):                                             # multiset of matrices per size unchanged
        m = np.where(n_lig == n)[0]
        assert sorted(labels[i].tobytes() for i in m) == sorted(new[i].tobytes() for i in m)
    assert sorted(src) == list(range(len(n_lig)))                          # a true permutation


def test_scramble_leaves_no_fixed_points_in_groups_of_two_or_more_and_breaks_alignment():
    rng = np.random.RandomState(2)
    n_lig = rng.randint(8, 14, 600)
    src = scramble_permutation(n_lig, seed=3)
    sizes = {n: (n_lig == n).sum() for n in np.unique(n_lig)}
    for i in range(len(n_lig)):
        if sizes[n_lig[i]] >= 2:
            assert src[i] != i
    # geometry->label link destroyed: label matrix of complex i no longer its own (checked via source index)
    assert (src != np.arange(len(src))).mean() > 0.99


def test_scramble_is_seed_deterministic_and_seed_sensitive():
    n_lig = np.random.RandomState(4).randint(8, 14, 300)
    assert np.array_equal(scramble_permutation(n_lig, 5), scramble_permutation(n_lig, 5))
    assert not np.array_equal(scramble_permutation(n_lig, 5), scramble_permutation(n_lig, 6))


def test_singleton_group_is_left_unscrambled():
    n_lig = np.array([10, 10, 10, 25])
    src = scramble_permutation(n_lig, 0)
    assert src[3] == 3 and all(src[i] != i for i in range(3))


if __name__ == '__main__':  # dependency-free runner (pytest also works if installed)
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith('test_') and callable(v)]
    for k, f in fns:
        f()
        print('PASS', k)
    print(f'{len(fns)} tests passed')
