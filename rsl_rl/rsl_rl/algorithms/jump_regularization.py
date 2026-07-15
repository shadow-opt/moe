import torch


class JumpMirror:
    """Name-derived sagittal mirror for WIN's 48-d observation and actions."""

    def __init__(self, dof_names, device):
        if len(dof_names) != 12:
            raise ValueError(f"Expected 12 dof names, got {len(dof_names)}: {dof_names}")
        name_to_index = {name: index for index, name in enumerate(dof_names)}
        if len(name_to_index) != len(dof_names):
            raise ValueError(f"Duplicate dof names: {dof_names}")

        joint_source = []
        joint_sign = []
        side_pairs = {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"}
        for name in dof_names:
            side, remainder = name.split("_", 1)
            counterpart = f"{side_pairs[side]}_{remainder}"
            if counterpart not in name_to_index:
                raise ValueError(f"Missing mirror counterpart {counterpart} for {name}")
            joint_source.append(name_to_index[counterpart])
            joint_sign.append(-1.0 if "hip" in remainder else 1.0)

        obs_source = list(range(48))
        obs_sign = [1.0] * 48
        obs_sign[0:3] = [-1.0, 1.0, -1.0]
        obs_sign[3:6] = [1.0, -1.0, 1.0]
        obs_sign[6:12] = [1.0, -1.0, -1.0, 1.0, 1.0, 1.0]
        for start in (12, 24, 36):
            for target, source in enumerate(joint_source):
                obs_source[start + target] = start + source
                obs_sign[start + target] = joint_sign[target]

        self.obs_source = torch.tensor(obs_source, dtype=torch.long, device=device)
        self.obs_sign = torch.tensor(obs_sign, dtype=torch.float, device=device)
        self.action_source = torch.tensor(joint_source, dtype=torch.long, device=device)
        self.action_sign = torch.tensor(joint_sign, dtype=torch.float, device=device)

    def observations(self, observations):
        if observations.shape[-1] != 48:
            raise ValueError(f"Expected observation dim 48, got {observations.shape}")
        return observations[..., self.obs_source] * self.obs_sign

    def history(self, history):
        if history.shape[-1] % 48 != 0:
            raise ValueError(f"History last dim must be a multiple of 48, got {history.shape}")
        if history.dim() == 2:
            shaped = history.reshape(history.shape[0], -1, 48)
            return self.observations(shaped).reshape_as(history)
        if history.shape[-1] == 48:
            return self.observations(history)
        raise ValueError(f"Unsupported history shape: {history.shape}")

    def actions(self, actions):
        if actions.shape[-1] != 12:
            raise ValueError(f"Expected action dim 12, got {actions.shape}")
        return actions[..., self.action_source] * self.action_sign
