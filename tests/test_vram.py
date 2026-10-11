from unittest import TestCase

from transcriber.vram import (
    CONTEXT_OVERHEAD_MIB,
    MAXIMUM_AUTOMATIC_BATCH,
    SAFETY_MARGIN_MIB,
    describe_allocation_failure,
    model_weights_mib,
    plan_batch_size,
    slot_cost_mib,
)


class EstimateTests(TestCase):
    def test_int8_halves_the_weight_footprint(self):
        self.assertEqual(
            model_weights_mib("large-v3", "int8_float16"),
            model_weights_mib("large-v3", "float16") // 2,
        )

    def test_float32_doubles_the_weight_footprint(self):
        self.assertEqual(
            model_weights_mib("turbo", "float32"),
            model_weights_mib("turbo", "float16") * 2,
        )

    def test_turbo_costs_as_much_per_slot_as_large_because_of_the_encoder(self):
        self.assertEqual(
            slot_cost_mib("turbo", "float16"), slot_cost_mib("large-v3", "float16")
        )

    def test_unknown_model_has_no_estimate(self):
        self.assertIsNone(model_weights_mib("./modelo-convertido", "float16"))
        self.assertIsNone(slot_cost_mib("./modelo-convertido", "float16"))


class PlanBatchSizeTests(TestCase):
    def _plan(self, requested, free_mib, **kwargs):
        options = {
            "model": "large-v3",
            "compute_type": "float16",
            "explicit": False,
        }
        options.update(kwargs)
        return plan_batch_size(requested, free_mib=free_mib, **options)

    def test_does_not_grow_when_memory_is_plentiful(self):
        # Measured: batch 32 took twice as long as batch 4 on the same audio, so free
        # VRAM is not a reason to raise the batch.
        plan = self._plan(4, 15000)

        self.assertEqual(plan.batch_size, 4)
        self.assertTrue(plan.fits)

    def test_caps_a_profile_asking_for_more_than_the_measured_optimum(self):
        plan = self._plan(32, 15000)

        self.assertEqual(plan.batch_size, MAXIMUM_AUTOMATIC_BATCH)

    def test_shrinks_when_another_workload_holds_the_card(self):
        plan = self._plan(8, 5000)

        self.assertLess(plan.batch_size, 8)
        self.assertTrue(plan.fits)
        self.assertIn("reduzir", plan.reason)

    def test_reports_that_the_model_does_not_fit(self):
        plan = self._plan(8, 1967, model="turbo")

        self.assertFalse(plan.fits)
        self.assertEqual(plan.batch_size, 1)
        self.assertIn("1967 MiB livres", plan.reason)

    def test_explicit_batch_is_never_overridden(self):
        for free_mib in (1000, 15000):
            plan = self._plan(6, free_mib, explicit=True)

            self.assertEqual(plan.batch_size, 6)
            self.assertIn("explicitamente", plan.reason)

    def test_keeps_the_profile_value_when_memory_is_unknown(self):
        plan = self._plan(8, None)

        self.assertEqual(plan.batch_size, 8)
        self.assertIn("desconhecida", plan.reason)

    def test_keeps_the_profile_value_for_an_uncatalogued_model(self):
        plan = self._plan(8, 15000, model="./modelo-convertido")

        self.assertEqual(plan.batch_size, 8)
        self.assertIn("não catalogado", plan.reason)

    def test_abundant_memory_never_raises_the_batch(self):
        plan = self._plan(4, 200000, model="tiny")

        self.assertEqual(plan.batch_size, 4)

    def test_reserves_context_and_safety_margin(self):
        weights = model_weights_mib("small", "float16")
        slot = slot_cost_mib("small", "float16")
        free = weights + CONTEXT_OVERHEAD_MIB + SAFETY_MARGIN_MIB + 3 * slot

        plan = self._plan(16, free, model="small")

        self.assertEqual(plan.batch_size, 3)

    def test_int8_sustains_the_profile_batch_where_float16_cannot(self):
        float16 = self._plan(8, 4200, compute_type="float16").batch_size
        int8 = self._plan(8, 4200, compute_type="int8_float16").batch_size

        self.assertGreater(int8, float16)


class AllocationFailureTests(TestCase):
    def test_names_the_occupancy_and_the_alternatives(self):
        message = describe_allocation_failure(
            free_mib=1967, model="turbo", compute_type="float16", device_total_mib=16311
        )

        self.assertIn("1967 MiB livres de 16311 MiB", message)
        self.assertIn("--compute-type int8_float16", message)
        self.assertIn("--device cpu", message)
        self.assertIn("libere a GPU", message)

    def test_works_without_memory_information(self):
        message = describe_allocation_failure(
            free_mib=None, model="turbo", compute_type="float16"
        )

        self.assertIn("turbo", message)
        self.assertNotIn("livres", message)
