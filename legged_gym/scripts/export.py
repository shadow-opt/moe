import argparse
import os
from typing import Dict, Optional, Tuple


import isaacgym
import torch
from legged_gym.envs import *
from legged_gym.utils import task_registry
from legged_gym.utils.exporter import export_policy_as_jit, export_policy_as_onnx
from legged_gym.utils.helpers import class_to_dict
import rsl_rl.modules as rsl_modules


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Export .pt (checkpoint or TorchScript) to TorchScript/ONNX")
	parser.add_argument("--input", type=str, required=True, help="Path to .pt file (training checkpoint or TorchScript)")
	parser.add_argument("--task", type=str, default=None, help="Task name used to rebuild model for checkpoint export")
	parser.add_argument("--output_dir", type=str, default=None, help="Directory to save exported files")
	parser.add_argument("--export", type=str, default="both", choices=["jit", "onnx", "both"], help="Export format")
	parser.add_argument("--jit_name", type=str, default="policy.pt", help="Output TorchScript filename")
	parser.add_argument("--onnx_name", type=str, default="policy.onnx", help="Output ONNX filename")
	parser.add_argument("--force_type", type=str, default="auto", choices=["auto", "checkpoint", "torchscript"], help="Force input .pt type")
	parser.add_argument("--input_dim", type=int, default=None, help="Observation dim for TorchScript->ONNX export")
	parser.add_argument("--opset", type=int, default=11, help="ONNX opset version for TorchScript->ONNX")
	parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device used for checkpoint reconstruction")
	parser.add_argument("--verbose", action="store_true", help="Verbose ONNX export")
	return parser.parse_args()


def _default_output_dir(input_path: str) -> str:
	return os.path.join(os.path.dirname(os.path.abspath(input_path)), "exported")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
	if not state_dict:
		return state_dict
	if all(key.startswith("module.") for key in state_dict.keys()):
		return {key[len("module."):]: value for key, value in state_dict.items()}
	return state_dict


def _try_load_torchscript(input_path: str) -> Optional[torch.jit.ScriptModule]:
	try:
		model = torch.jit.load(input_path, map_location="cpu")
		model.eval()
		return model
	except Exception:
		return None


def _extract_state_dict(loaded_obj) -> Optional[Dict[str, torch.Tensor]]:
	if not isinstance(loaded_obj, dict):
		return None
	if "model_state_dict" in loaded_obj and isinstance(loaded_obj["model_state_dict"], dict):
		return loaded_obj["model_state_dict"]
	if "state_dict" in loaded_obj and isinstance(loaded_obj["state_dict"], dict):
		return loaded_obj["state_dict"]
	if loaded_obj and all(torch.is_tensor(v) for v in loaded_obj.values()):
		return loaded_obj
	return None


def _build_policy_from_task(task: str, device: str):
	env_cfg, train_cfg = task_registry.get_cfgs(name=task)
	train_cfg_dict = class_to_dict(train_cfg)

	runner_cfg = train_cfg_dict["runner"]
	policy_cfg = train_cfg_dict["policy"]
	policy_class_name = runner_cfg["policy_class_name"]
	policy_class = getattr(rsl_modules, policy_class_name)

	num_actor_obs = env_cfg.env.num_observations
	num_critic_obs = env_cfg.env.num_privileged_obs
	if num_critic_obs is None:
		num_critic_obs = num_actor_obs
	num_actions = env_cfg.env.num_actions

	if train_cfg.runner_class_name == "OnPolicyRunnerCTS":
		model = policy_class(
			num_actor_obs,
			num_critic_obs,
			num_actions,
			1,
			train_cfg.history_length,
			**policy_cfg,
		).to(device)
	else:
		model = policy_class(
			num_actor_obs,
			num_critic_obs,
			num_actions,
			**policy_cfg,
		).to(device)
	model.eval()
	return model, num_actor_obs


def _export_torchscript_input(
	script_model: torch.jit.ScriptModule,
	output_dir: str,
	export_jit: bool,
	export_onnx: bool,
	jit_name: str,
	onnx_name: str,
	input_dim: Optional[int],
	task: Optional[str],
	opset: int,
	verbose: bool,
) -> None:
	os.makedirs(output_dir, exist_ok=True)

	if export_jit:
		script_model.save(os.path.join(output_dir, jit_name))

	if export_onnx:
		obs_dim = input_dim
		is_cts_task = False
		if obs_dim is None and task is not None:
			env_cfg, train_cfg = task_registry.get_cfgs(name=task)
			obs_dim = env_cfg.env.num_observations
			is_cts_task = (getattr(train_cfg, "runner_class_name", "") == "OnPolicyRunnerCTS")
		if obs_dim is None:
			raise ValueError("TorchScript -> ONNX 导出需要输入维度，请提供 --input_dim 或 --task。")
		print(f"[INFO] Exporting TorchScript -> ONNX with single-frame obs dim: {obs_dim}")
		print("[INFO] This ONNX uses non-stacked actor observations (e.g., 45 for go2).")
		print("[WARN] TorchScript -> ONNX is not recommended as final deployment format for CTS/MoE policies.")
		print("[WARN] Reason: ONNX runtime calls are stateless by default, while CTS/MoE TorchScript relies on temporal history behavior.")
		if is_cts_task:
			print("[WARN] Detected task runner: OnPolicyRunnerCTS.")
		print("[WARN] For deployment, prefer checkpoint -> ONNX export (stacked-history input) to match training-time observation semantics.")

		scripted_cpu = script_model.to("cpu").eval()
		dummy_obs = torch.zeros(1, obs_dim, dtype=torch.float32)
		torch.onnx.export(
			scripted_cpu,
			dummy_obs,
			os.path.join(output_dir, onnx_name),
			export_params=True,
			opset_version=opset,
			verbose=verbose,
			input_names=["obs"],
			dynamic_axes={},
		)


def _export_checkpoint_input(
	input_path: str,
	output_dir: str,
	export_jit: bool,
	export_onnx: bool,
	jit_name: str,
	onnx_name: str,
	task: str,
	device: str,
	verbose: bool,
) -> None:
	model, _ = _build_policy_from_task(task, device)

	loaded_obj = torch.load(input_path, map_location=device)
	state_dict = _extract_state_dict(loaded_obj)
	if state_dict is None:
		raise ValueError("输入 .pt 不是有效 checkpoint（未找到 state_dict/model_state_dict）。")
	state_dict = _strip_module_prefix(state_dict)

	missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
	if missing_keys or unexpected_keys:
		print(f"[WARN] missing_keys: {len(missing_keys)}, unexpected_keys: {len(unexpected_keys)}")
		if missing_keys:
			print(f"[WARN] first missing keys: {missing_keys[:10]}")
		if unexpected_keys:
			print(f"[WARN] first unexpected keys: {unexpected_keys[:10]}")

	os.makedirs(output_dir, exist_ok=True)
	if export_jit:
		export_policy_as_jit(model, output_dir, filename=jit_name)
	if export_onnx:
		expected_onnx_obs_dim = None
		if hasattr(model, "actor") and hasattr(model.actor, "num_obs_history"):
			expected_onnx_obs_dim = int(model.actor.num_obs_history)
		if expected_onnx_obs_dim is not None:
			print(f"[INFO] Exporting checkpoint -> ONNX with history-stacked obs dim: {expected_onnx_obs_dim}")
			print("[INFO] This ONNX is compatible with deploy_mujoco/deploy_onnx.py stacked-history input.")
		export_policy_as_onnx(model, output_dir, filename=onnx_name, verbose=verbose)


def _resolve_input_type(input_path: str, force_type: str) -> Tuple[str, Optional[torch.jit.ScriptModule]]:
	if force_type == "torchscript":
		scripted = _try_load_torchscript(input_path)
		if scripted is None:
			raise ValueError("--force_type=torchscript 但文件无法被 torch.jit.load 加载。")
		return "torchscript", scripted

	if force_type == "checkpoint":
		return "checkpoint", None

	scripted = _try_load_torchscript(input_path)
	if scripted is not None:
		return "torchscript", scripted
	return "checkpoint", None


def main():
	args = parse_args()

	input_path = os.path.abspath(args.input)
	if not os.path.isfile(input_path):
		raise FileNotFoundError(f"Input file does not exist: {input_path}")
	if not input_path.endswith(".pt"):
		raise ValueError("输入文件必须是 .pt")

	output_dir = os.path.abspath(args.output_dir) if args.output_dir else _default_output_dir(input_path)
	export_jit = args.export in ("jit", "both")
	export_onnx = args.export in ("onnx", "both")

	input_type, scripted_model = _resolve_input_type(input_path, args.force_type)
	print(f"[INFO] Input type resolved as: {input_type}")
	print(f"[INFO] Output directory: {output_dir}")

	if input_type == "torchscript":
		_export_torchscript_input(
			script_model=scripted_model,
			output_dir=output_dir,
			export_jit=export_jit,
			export_onnx=export_onnx,
			jit_name=args.jit_name,
			onnx_name=args.onnx_name,
			input_dim=args.input_dim,
			task=args.task,
			opset=args.opset,
			verbose=args.verbose,
		)
	else:
		if args.task is None:
			raise ValueError("输入是 checkpoint 时必须提供 --task 来重建模型结构。")
		_export_checkpoint_input(
			input_path=input_path,
			output_dir=output_dir,
			export_jit=export_jit,
			export_onnx=export_onnx,
			jit_name=args.jit_name,
			onnx_name=args.onnx_name,
			task=args.task,
			device=args.device,
			verbose=args.verbose,
		)

	print("[INFO] Export finished.")
	if export_jit:
		print(f"[INFO] TorchScript: {os.path.join(output_dir, args.jit_name)}")
	if export_onnx:
		print(f"[INFO] ONNX: {os.path.join(output_dir, args.onnx_name)}")


if __name__ == "__main__":
	main()
