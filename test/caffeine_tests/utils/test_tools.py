import numpy as np

from caffeine.utils.common_tools import gen_module_id, bcl_repr, get_class_weight


def test_bcl_repr():
    labels = np.array([[1, 0, 1, 1]])
    labels = bcl_repr(labels)
    assert np.all(np.equal(labels, np.array([1, 0, 1, 1])))


def test_get_class_weight():
    labels = np.array([1, 1, 2, 2, 3, 3, 3, 4])
    class_weights = get_class_weight(labels)
    print(class_weights)
    assert class_weights == {
        1: 1.0,
        2: 1.0, 
        3: 2.0/3,
        4: 2.0}


def test_module_id():
    assert gen_module_id("123")[:4] == "123-"
