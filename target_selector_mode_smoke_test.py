import unittest

import numpy as np

from Lyapunov.target_selector import (
    HAS_CVXPY,
    REFINED_STEP_A_SELECTOR_NAME,
    build_target_selector_config,
    prepare_filter_target,
)


def _synthetic_problem():
    A_aug = np.array(
        [
            [0.7, 0.1, 0.2],
            [0.0, 0.8, 0.1],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    B_aug = np.array([[1.0], [0.3], [0.0]], dtype=float)
    C_aug = np.array([[1.0, 0.2, 1.0]], dtype=float)
    xhat_aug = np.array([0.0, 0.0, 0.25], dtype=float)
    u_min = np.array([-0.5], dtype=float)
    u_max = np.array([0.5], dtype=float)
    return A_aug, B_aug, C_aug, xhat_aug, u_min, u_max


@unittest.skipUnless(HAS_CVXPY, "cvxpy is required for target selector smoke tests")
class RefinedStepATargetSelectorSmokeTest(unittest.TestCase):
    def test_returns_standard_schema(self):
        A_aug, B_aug, C_aug, xhat_aug, u_min, u_max = _synthetic_problem()
        cfg = build_target_selector_config(
            n_x=2,
            n_u=1,
            n_y=1,
            n_d=1,
            Q_out=np.ones(1),
            Rmove_diag=np.ones(1),
        )
        info = prepare_filter_target(
            A_aug=A_aug,
            B_aug=B_aug,
            C_aug=C_aug,
            xhat_aug=xhat_aug,
            y_sp=np.array([0.1]),
            u_min=u_min,
            u_max=u_max,
            u_applied_k=np.array([0.2]),
            config=cfg,
        )
        self.assertIn("selector_name", info)
        self.assertEqual(info["selector_name"], REFINED_STEP_A_SELECTOR_NAME)
        self.assertIn("objective_terms", info)
        if info["success"]:
            self.assertEqual(info["x_s"].shape, (2,))
            self.assertEqual(info["u_s"].shape, (1,))
            self.assertEqual(info["d_s"].shape, (1,))

    def test_previous_target_terms_activate(self):
        A_aug, B_aug, C_aug, xhat_aug, u_min, u_max = _synthetic_problem()
        cfg = build_target_selector_config(
            n_x=2,
            n_u=1,
            n_y=1,
            n_d=1,
            Q_out=np.ones(1),
            Rmove_diag=np.ones(1),
        )
        prev_target = {
            "success": True,
            "x_s": np.array([0.1, -0.1]),
            "u_s": np.array([0.05]),
        }
        info = prepare_filter_target(
            A_aug=A_aug,
            B_aug=B_aug,
            C_aug=C_aug,
            xhat_aug=xhat_aug,
            y_sp=np.array([0.1]),
            u_min=u_min,
            u_max=u_max,
            u_applied_k=np.array([0.2]),
            config=cfg,
            prev_target=prev_target,
        )
        dbg = info.get("selector_debug", {})
        self.assertTrue(bool(dbg.get("prev_input_term_active", False)))
        self.assertTrue(bool(dbg.get("prev_state_term_active", False)))

    def test_h_matrix_path_returns_controlled_reference(self):
        A_aug, B_aug, C_aug, xhat_aug, u_min, u_max = _synthetic_problem()
        C_aug = np.array(
            [
                [1.0, 0.2, 1.0],
                [0.4, 0.8, 0.0],
            ],
            dtype=float,
        )
        A_aug = np.block(
            [
                [A_aug[:2, :2], np.array([[0.2, 0.0], [0.1, 0.1]], dtype=float)],
                [np.zeros((2, 2), dtype=float), np.eye(2, dtype=float)],
            ]
        )
        B_aug = np.array([[1.0], [0.3], [0.0], [0.0]], dtype=float)
        xhat_aug = np.array([0.0, 0.0, 0.25, -0.1], dtype=float)
        H = np.array([[1.0, 0.0]], dtype=float)
        cfg = build_target_selector_config(
            n_x=2,
            n_u=1,
            n_y=2,
            n_d=2,
            Q_out=np.ones(2),
            Rmove_diag=np.ones(1),
        )
        info = prepare_filter_target(
            A_aug=A_aug,
            B_aug=B_aug,
            C_aug=C_aug,
            xhat_aug=xhat_aug,
            y_sp=np.array([0.2]),
            u_min=u_min,
            u_max=u_max,
            u_applied_k=np.array([0.1]),
            config=cfg,
            H=H,
        )
        if info["success"]:
            self.assertEqual(info["r_s"].shape, (1,))


if __name__ == "__main__":
    unittest.main()
