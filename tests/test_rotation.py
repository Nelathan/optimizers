import unittest

from sumotrack import RoundRobinRefreshScheduler


class RoundRobinRefreshSchedulerTest(unittest.TestCase):
    def test_budget_one_walks_stable_order(self):
        scheduler = RoundRobinRefreshScheduler(num_items=3, subspace_refresh_budget=1)

        self.assertEqual(scheduler.next_refresh_indices(), (0,))
        self.assertEqual(scheduler.next_refresh_indices(), (1,))
        self.assertEqual(scheduler.next_refresh_indices(), (2,))
        self.assertEqual(scheduler.next_refresh_indices(), (0,))

    def test_budget_two_wraps_explicitly(self):
        scheduler = RoundRobinRefreshScheduler(num_items=3, subspace_refresh_budget=2)

        self.assertEqual(scheduler.next_refresh_indices(), (0, 1))
        self.assertEqual(scheduler.next_refresh_indices(), (2, 0))
        self.assertEqual(scheduler.next_refresh_indices(), (1, 2))

    def test_target_interval_derives_budget(self):
        scheduler = RoundRobinRefreshScheduler.from_target_interval(num_items=10, target_refresh_interval=4)

        self.assertEqual(scheduler.subspace_refresh_budget, 3)
        self.assertEqual(scheduler.next_refresh_indices(), (0, 1, 2))

    def test_refresh_is_separate_from_ordinary_updates(self):
        scheduler = RoundRobinRefreshScheduler(num_items=4, subspace_refresh_budget=1)

        refresh = scheduler.next_refresh_indices()
        ordinary_updates = scheduler.all_update_indices()

        self.assertEqual(refresh, (0,))
        self.assertEqual(ordinary_updates, (0, 1, 2, 3))

    def test_empty_scheduler_is_stable(self):
        scheduler = RoundRobinRefreshScheduler(num_items=0, subspace_refresh_budget=2)

        self.assertEqual(scheduler.next_refresh_indices(), ())
        self.assertEqual(scheduler.all_update_indices(), ())


if __name__ == "__main__":
    unittest.main()
