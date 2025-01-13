from surfwetter_ml import compute


def test_compute():
    assert compute(["a", "bc", "abc"]) == "abc"
