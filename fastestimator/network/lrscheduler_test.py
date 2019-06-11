from fastestimator.network.lrscheduler import CyclicScheduler
import numpy as np

def test_cyclic_lr1():
    scheduler = CyclicScheduler(num_cycle=1, cycle_multiplier=1, decrease_method="cosine")
    scheduler.epochs = 50
    scheduler.steps_per_epoch = 100
    assert scheduler.lr_schedule_fn(0) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(2500), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(5000), 1e-6)

def test_cyclic_lr2():
    scheduler = CyclicScheduler(num_cycle=2, cycle_multiplier=2, decrease_method="cosine")
    scheduler.epochs = 30
    scheduler.steps_per_epoch = 300
    assert scheduler.lr_schedule_fn(0) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(1500), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(2999), 1e-6, decimal=4)
    assert scheduler.lr_schedule_fn(3000) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(6000), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(9000), 1e-6)

def test_cyclic_lr3():
    scheduler = CyclicScheduler(num_cycle=3, cycle_multiplier=2, decrease_method="cosine")
    scheduler.epochs = 7
    scheduler.steps_per_epoch = 1000
    assert scheduler.lr_schedule_fn(0) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(500), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(999), 1e-6, decimal=4)
    assert scheduler.lr_schedule_fn(1000) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(2000), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(2999), 1e-6, decimal=4)
    assert scheduler.lr_schedule_fn(3000) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(5000), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(7000), 1e-6)

def test_cyclic_lr4():
    scheduler = CyclicScheduler(num_cycle=3, cycle_multiplier=1, decrease_method="cosine")
    scheduler.epochs = 9
    scheduler.steps_per_epoch = 1000
    assert scheduler.lr_schedule_fn(0) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(1500), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(2999), 1e-6, decimal=4)
    assert scheduler.lr_schedule_fn(3000) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(4500), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(5999), 1e-6, decimal=4)
    assert scheduler.lr_schedule_fn(6000) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(7500), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(9000), 1e-6)

def test_cyclic_lr5():
    scheduler = CyclicScheduler(num_cycle=1, cycle_multiplier=1, decrease_method="linear")
    scheduler.epochs = 50
    scheduler.steps_per_epoch = 100
    assert scheduler.lr_schedule_fn(0) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(2500), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(5000), 1e-6)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(5000) - scheduler.lr_schedule_fn(4000),
                                   scheduler.lr_schedule_fn(4000) - scheduler.lr_schedule_fn(3000))

def test_cyclic_lr6():
    scheduler = CyclicScheduler(num_cycle=2, cycle_multiplier=2, decrease_method="linear")
    scheduler.epochs = 30
    scheduler.steps_per_epoch = 300
    assert scheduler.lr_schedule_fn(0) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(1500), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(2999), 1e-6, decimal=3)
    assert scheduler.lr_schedule_fn(3000) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(6000), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(9000), 1e-6)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(5000) - scheduler.lr_schedule_fn(4000),
                                   scheduler.lr_schedule_fn(4000) - scheduler.lr_schedule_fn(3000))

def test_cyclic_lr7():
    scheduler = CyclicScheduler(num_cycle=3, cycle_multiplier=2, decrease_method="linear")
    scheduler.epochs = 7
    scheduler.steps_per_epoch = 1000
    assert scheduler.lr_schedule_fn(0) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(500), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(999), 1e-6, decimal=3)
    assert scheduler.lr_schedule_fn(1000) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(2000), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(2999), 1e-6, decimal=3)
    assert scheduler.lr_schedule_fn(3000) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(5000), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(7000), 1e-6)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(5000) - scheduler.lr_schedule_fn(4000),
                                   scheduler.lr_schedule_fn(4000) - scheduler.lr_schedule_fn(3000))

def test_cyclic_lr8():
    scheduler = CyclicScheduler(num_cycle=3, cycle_multiplier=1, decrease_method="linear")
    scheduler.epochs = 9
    scheduler.steps_per_epoch = 1000
    assert scheduler.lr_schedule_fn(0) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(1500), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(2999), 1e-6, decimal=3)
    assert scheduler.lr_schedule_fn(3000) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(4500), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(5999), 1e-6, decimal=3)
    assert scheduler.lr_schedule_fn(6000) == 1.0
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(7500), (1e-6 + 1)/2)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(9000), 1e-6)
    np.testing.assert_almost_equal(scheduler.lr_schedule_fn(5000) - scheduler.lr_schedule_fn(4000),
                                   scheduler.lr_schedule_fn(4000) - scheduler.lr_schedule_fn(3000))