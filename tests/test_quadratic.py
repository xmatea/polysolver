import pytest
import numpy as np
from polysolve.polysolve import quadratic

@pytest.mark.parameterize("params , expected", [([1, 0, 0], [0, 0])])
def test_quadratic(params, expected):
    """ tests """
    res = quadratic(*params)
    assert all(np.isclose(np.polyval(params, root), expected[i]) for i, root in enumerate(res))


@pytest.mark.parameterize("params , expected", [([1, 0, 0], [0, 0])])
def test_quadratic_fails(params, expected):
    """ tests """
    res = quadratic(*params)
    with pytest.raises(ValueError, match="negative discriminant"):

    assert all(np.isclose(np.polyval(params, root), 0) for root in res)