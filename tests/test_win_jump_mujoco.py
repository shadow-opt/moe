import importlib.util
import sys
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).parents[1]
MODULE_PATH = ROOT / "deploy" / "deploy_mujoco" / "deploy_win_jump_moe_cts.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("deploy_win_jump_moe_cts", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_z2_scene_has_expected_joint_and_foot_interface():
    model = mujoco.MjModel.from_xml_path(
        str(ROOT / "resources" / "robots" / "aaaaa_fixed" / "z2" / "urdf" / "scene_terrain.xml")
    )
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
    ]
    assert len(joint_names) == 12
    assert model.nu == 12
    for name in ("FL_foot", "RL_foot", "FR_foot", "RR_foot"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) >= 0


def test_quaternion_inverse_rotation_is_identity_at_neutral_pose():
    vector = np.asarray([1.0, 2.0, 3.0])
    np.testing.assert_allclose(MODULE.rotate_inverse([1.0, 0.0, 0.0, 0.0], vector), vector)


def test_joint_name_mapping_preserves_one_hot_joint_identity():
    model_names = ["FL", "FR", "RL", "RR"]
    qpos_names = ["FL", "RL", "FR", "RR"]
    actuator_names = ["RL", "FL", "RR", "FR"]
    model_from_qpos, actuator_from_model = MODULE.build_joint_maps(
        model_names, qpos_names, actuator_names
    )
    qpos_values = np.asarray([1.0, 3.0, 2.0, 4.0])
    model_values = qpos_values[model_from_qpos]
    np.testing.assert_array_equal(model_values, [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(model_values[actuator_from_model], [3.0, 1.0, 4.0, 2.0])


def test_new_config_is_not_the_legacy_470_dim_runner():
    config_path = ROOT / "deploy" / "deploy_mujoco" / "configs" / "win_jump_moe_cts.yaml"
    text = config_path.read_text(encoding="utf-8")
    assert "manifest_path" in text
    assert "scene_terrain.xml" in text
    assert "470" not in text
