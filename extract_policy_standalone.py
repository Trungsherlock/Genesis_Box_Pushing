"""Standalone script to extract Go2 policy network to ONNX without Genesis dependencies"""
import argparse
import os
import pickle
import torch
import torch.nn as nn
from collections import OrderedDict


class MLP(nn.Module):
    """Multi-Layer Perceptron matching RSL-RL's ActorCritic.actor structure"""
    def __init__(self, input_dim, hidden_dims, output_dim, activation='elu'):
        super().__init__()
        
        # Build the network layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Add activation except for the last layer
            if i < len(dims) - 2:
                if activation == 'elu':
                    layers.append(nn.ELU())
                elif activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def extract_actor_weights(checkpoint_path):
    """Extract just the actor network weights from checkpoint"""
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract actor weights
    actor_state_dict = OrderedDict()
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('actor.'):
            # Remove 'actor.' prefix and rename to match Sequential module
            new_key = key.replace('actor.', '')
            actor_state_dict[new_key] = value
    
    # Also get the action std if available
    if 'std' in checkpoint['model_state_dict']:
        std = checkpoint['model_state_dict']['std']
        print(f"  - Action std shape: {std.shape}")
    else:
        std = None
    
    print(f"  - Found {len(actor_state_dict)} actor layers")
    return actor_state_dict, std, checkpoint.get('iter', 0)


def create_policy_network(obs_dim, action_dim, hidden_dims, activation='elu'):
    """Create the policy network architecture"""
    print(f"\n🏗️ Creating policy network:")
    print(f"  - Input dimension: {obs_dim}")
    print(f"  - Hidden dimensions: {hidden_dims}")
    print(f"  - Output dimension: {action_dim}")
    print(f"  - Activation: {activation}")
    
    policy = MLP(obs_dim, hidden_dims, action_dim, activation)
    return policy


def export_to_onnx(model, obs_dim, output_path, std=None):
    """Export the policy network to ONNX format with comprehensive documentation"""
    print(f"\n📦 Exporting to ONNX: {output_path}")
    
    # Create comprehensive documentation string for the ONNX model
    doc_string = """Go2 Quadruped Locomotion Policy Network

OBSERVATION SPACE (45 dimensions):
┌─────────────────────────────────────────────────────────────────────────────┐
│ Index │ Component          │ Dim │ Description                    │ Units   │
├─────────────────────────────────────────────────────────────────────────────┤
│ [0:3] │ Angular Velocity   │  3  │ Base angular velocity in       │ rad/s   │
│       │                    │     │ robot frame (roll, pitch, yaw) │         │
├─────────────────────────────────────────────────────────────────────────────┤
│ [3:6] │ Projected Gravity  │  3  │ Gravity vector projected into  │ unit vec│
│       │                    │     │ robot frame (for orientation)  │         │
├─────────────────────────────────────────────────────────────────────────────┤
│ [6:9] │ Command Velocities │  3  │ Desired velocities:            │ m/s,    │
│       │                    │     │ [vx, vy, yaw_rate]             │ rad/s   │
├─────────────────────────────────────────────────────────────────────────────┤
│ [9:21]│ Joint Positions    │ 12  │ Current joint angles relative  │ rad     │
│       │                    │     │ to default pose                │         │
├─────────────────────────────────────────────────────────────────────────────┤
│[21:33]│ Joint Velocities   │ 12  │ Current joint angular          │ rad/s   │
│       │                    │     │ velocities                     │         │
├─────────────────────────────────────────────────────────────────────────────┤
│[33:45]│ Previous Actions   │ 12  │ Previous motor commands        │ rad     │
│       │                    │     │ (for temporal consistency)     │         │
└─────────────────────────────────────────────────────────────────────────────┘

Joint Order (for positions, velocities, actions):
[FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
 RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf]

ACTION SPACE (12 dimensions):
- Motor position commands (rad) relative to default pose
- Scaled by action_scale (0.25) in environment
- Applied as: target_pos = action * 0.25 + default_pose

SCALING FACTORS (applied in observation):
- Angular velocity: * 0.25
- Linear velocity commands: * 2.0
- Joint positions: * 1.0 (relative to default)
- Joint velocities: * 0.05

DEFAULT JOINT ANGLES (rad):
- Hip joints (FR, FL, RR, RL): 0.0
- Thigh joints (FR, FL): 0.8, (RR, RL): 1.0
- Calf joints (all): -1.5

CONTROL PARAMETERS:
- PD gains: kp=20.0, kd=0.5
- Control frequency: 50Hz (dt=0.02s)
- Action latency: 1 step (simulated)"""
    
    # Set model to eval mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, obs_dim)
    
    # Test forward pass
    with torch.no_grad():
        test_output = model(dummy_input)
        print(f"  - Test output shape: {test_output.shape}")
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observation'],
        output_names=['action_mean'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action_mean': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"✅ Exported ONNX model to: {output_path}")
    
    # Save std separately if available
    if std is not None:
        std_path = output_path.replace('.onnx', '_std.pt')
        torch.save({'std': std}, std_path)
        print(f"✅ Saved action std to: {std_path}")
    
    # Try to validate and add metadata if onnx is available
    try:
        import onnx
        from onnx import helper
        
        # Load the model
        onnx_model = onnx.load(output_path)
        
        # Add the comprehensive documentation string
        onnx_model.doc_string = doc_string
        
        # Add custom metadata properties
        metadata = [
            ("model_type", "quadruped_locomotion_policy"),
            ("robot", "Go2"),
            ("framework", "Genesis"),
            ("observation_dim", str(obs_dim)),
            ("action_dim", "12"),
            ("joint_order", "FR_hip,FR_thigh,FR_calf,FL_hip,FL_thigh,FL_calf,RR_hip,RR_thigh,RR_calf,RL_hip,RL_thigh,RL_calf"),
            ("observation_components", "ang_vel[0:3],proj_gravity[3:6],commands[6:9],joint_pos[9:21],joint_vel[21:33],prev_actions[33:45]"),
            ("scaling_ang_vel", "0.25"),
            ("scaling_lin_vel", "2.0"),
            ("scaling_joint_pos", "1.0"),
            ("scaling_joint_vel", "0.05"),
            ("action_scale", "0.25"),
            ("control_frequency", "50Hz"),
            ("pd_gains", "kp=20.0,kd=0.5"),
            ("default_pose", "hip=0.0,thigh_front=0.8,thigh_rear=1.0,calf=-1.5")
        ]
        
        for key, value in metadata:
            onnx_model.metadata_props.append(
                onnx.StringStringEntryProto(key=key, value=value)
            )
        
        # Add descriptions to input/output tensors
        onnx_model.graph.input[0].doc_string = "Robot observation vector (45D): sensor readings and state information"
        onnx_model.graph.output[0].doc_string = "Motor position commands (12D): target joint angles relative to default pose"
        
        # Save the updated model
        onnx.save(onnx_model, output_path)
        
        # Validate the model
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model validation passed")
        print(f"✅ Added {len(metadata)} metadata properties")
        
    except ImportError:
        print("⚠️  onnx not installed, skipping validation and metadata")
    except Exception as e:
        print(f"⚠️  ONNX validation/metadata failed: {e}")


def export_to_torchscript(model, obs_dim, output_path, std=None):
    """Export the policy network to TorchScript format"""
    print(f"\n📦 Exporting to TorchScript: {output_path}")
    
    # Set model to eval mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, obs_dim)
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save the model with metadata
    if std is not None:
        # Create a wrapper that includes std
        class PolicyWithStd(nn.Module):
            def __init__(self, policy, std):
                super().__init__()
                self.policy = policy
                self.register_buffer('std', std)
            
            def forward(self, x):
                return self.policy(x)
        
        full_model = PolicyWithStd(model, std)
        traced_full = torch.jit.trace(full_model, dummy_input)
        torch.jit.save(traced_full, output_path)
    else:
        torch.jit.save(traced_model, output_path)
    
    print(f"✅ Exported TorchScript model to: {output_path}")
    
    # Test loading
    loaded_model = torch.jit.load(output_path)
    loaded_model.eval()
    with torch.no_grad():
        test_output = loaded_model(dummy_input)
    print(f"✅ TorchScript test passed, output shape: {test_output.shape}")


def main():
    parser = argparse.ArgumentParser(description="Extract Go2 policy to ONNX/TorchScript")
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking",
                        help="Experiment name")
    parser.add_argument("--ckpt", type=int, default=100,
                        help="Checkpoint number")
    parser.add_argument("--format", type=str, default="both", 
                        choices=["onnx", "torchscript", "both"],
                        help="Export format")
    parser.add_argument("--output_dir", type=str, default="./exported_models",
                        help="Output directory")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🤖 Go2 Policy Network Extractor (Standalone)")
    print("=" * 60)
    print(f"Experiment: {args.exp_name}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Export format: {args.format}")
    print("=" * 60)
    
    # Load configs to get dimensions
    cfg_path = f"logs/{args.exp_name}/cfgs.pkl"
    if not os.path.exists(cfg_path):
        print(f"❌ Config file not found: {cfg_path}")
        return
    
    print(f"\n📄 Loading configs from: {cfg_path}")
    with open(cfg_path, 'rb') as f:
        env_cfg, obs_cfg, _, _, train_cfg = pickle.load(f)
    
    # Extract dimensions
    obs_dim = obs_cfg['num_obs']
    action_dim = env_cfg['num_actions']
    
    # Default hidden dimensions if not in config
    if isinstance(train_cfg, dict) and 'policy' in train_cfg:
        hidden_dims = train_cfg['policy'].get('hidden_dims', [512, 256, 128])
        activation = train_cfg['policy'].get('activation', 'elu')
    else:
        # Default architecture based on the checkpoint we inspected
        hidden_dims = [512, 256, 128]
        activation = 'elu'
    
    print(f"  - Observation dimension: {obs_dim}")
    print(f"  - Action dimension: {action_dim}")
    print(f"  - Hidden dimensions: {hidden_dims}")
    
    # Create policy network
    policy = create_policy_network(obs_dim, action_dim, hidden_dims, activation)
    
    # Load checkpoint weights
    checkpoint_path = f"logs/{args.exp_name}/model_{args.ckpt}.pt"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    actor_weights, std, iteration = extract_actor_weights(checkpoint_path)
    print(f"  - Checkpoint iteration: {iteration}")
    
    # Load weights into policy network
    policy.net.load_state_dict(actor_weights)
    print("✅ Loaded actor weights into policy network")
    
    # Export to desired formats
    base_name = f"{args.exp_name}_policy_iter{iteration}"
    
    if args.format in ["onnx", "both"]:
        onnx_path = os.path.join(args.output_dir, f"{base_name}.onnx")
        export_to_onnx(policy, obs_dim, onnx_path, std)
    
    if args.format in ["torchscript", "both"]:
        ts_path = os.path.join(args.output_dir, f"{base_name}.pt")
        export_to_torchscript(policy, obs_dim, ts_path, std)
    
    # Print usage instructions
    print("\n" + "=" * 60)
    print("🎉 Export complete!")
    print("=" * 60)
    print("\n📝 Usage examples:")
    
    if args.format in ["onnx", "both"]:
        print("\n### Reading ONNX Model Documentation:")
        print("```python")
        print("import onnx")
        print(f"model = onnx.load('{base_name}.onnx')")
        print("# Print comprehensive documentation")
        print("print(model.doc_string)")
        print("# Print metadata properties")
        print("for prop in model.metadata_props:")
        print("    print(f'{prop.key}: {prop.value}')")
        print("```")
        print("")
        print("### Python ONNX inference:")
        print("```python")
        print("import onnxruntime as ort")
        print("import numpy as np")
        print("")
        print(f"# Load model")
        print(f"session = ort.InferenceSession('{base_name}.onnx')")
        print("")
        print(f"# Create observation (shape: {obs_dim})")
        print("# Components: [ang_vel(3), gravity(3), commands(3), joint_pos(12), joint_vel(12), prev_actions(12)]")
        print(f"obs = np.random.randn(1, {obs_dim}).astype(np.float32)")
        print("")
        print("# Run inference")
        print("action_mean = session.run(None, {'observation': obs})[0]")
        print(f"# Output shape: (1, {action_dim}) - motor position commands")
        print("```")
    
    if args.format in ["torchscript", "both"]:
        print("\n### Python TorchScript inference:")
        print("```python")
        print("import torch")
        print("")
        print(f"# Load model")
        print(f"model = torch.jit.load('{base_name}.pt')")
        print("model.eval()")
        print("")
        print(f"# Create observation (shape: {obs_dim})")
        print(f"obs = torch.randn(1, {obs_dim})")
        print("")
        print("# Run inference")
        print("with torch.no_grad():")
        print("    action_mean = model(obs)")
        if std is not None:
            print("    action_std = model.std  # If using PolicyWithStd wrapper")
        print(f"# Output shape: (1, {action_dim})")
        print("```")
    
    print("\n### C++ Integration:")
    print("Both ONNX and TorchScript models can be loaded in C++ for deployment.")
    print("- ONNX: Use ONNX Runtime C++ API")
    print("- TorchScript: Use LibTorch C++ API")


if __name__ == "__main__":
    main()