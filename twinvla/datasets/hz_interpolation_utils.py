import math
import torch
import numpy as np
import scipy.spatial.transform as spt

# Interpolation utility functions
def rot6d_to_matrix(rot_6d):
    """Convert 6D rotation representation to rotation matrix.
    
    Args:
        rot_6d: 6D rotation representation, shape [2, 3]
    
    Returns:
        Rotation matrix, shape [3, 3]
    """
    a1 = rot_6d[:3]  # First 3 elements
    a2 = rot_6d[3:6]  # Last 3 elements
    
    # Add small epsilon for numerical stability
    eps = 1e-7
    
    # Normalize first vector
    b1 = a1 / (torch.norm(a1, dim=0, keepdim=True) + eps)
    
    # Get second vector that's orthogonal to the first
    c = torch.cross(b1, a2, dim=0)
    c = c / (torch.norm(c, dim=0, keepdim=True) + eps)
    
    # Third vector orthogonal to first two
    b2 = torch.cross(c, b1)
    
    # Combine to form rotation matrix
    rot_matrix = torch.stack([b1, b2, c], dim=1)
    return rot_matrix

def matrix_to_rot6d(rot_matrix):
    """Convert rotation matrix to 6D rotation representation.
    
    Args:
        rot_matrix: Rotation matrix, shape [3, 3]
    
    Returns:
        6D rotation representation, shape [6]
    """
    # Extract first two columns of the rotation matrix to create 6D representation
    col1 = rot_matrix[:, 0]  # (3,)
    col2 = rot_matrix[:, 1]  # (3,)
    rotation_6d = torch.cat([col1, col2], dim=0)  # (6,)
    
    return rotation_6d

def interpolate_rotations(rotation_matrices, t_original, t_target):
    """Interpolate rotation matrices using quaternion SLERP.
    
    Args:
        rotation_matrices: List of rotation matrices
        t_original: Original time points
        t_target: Target time points
    
    Returns:
        List of interpolated rotation matrices
    """
    
    # Convert rotation matrices to quaternions
    rots = spt.Rotation.from_matrix(np.stack([mat.numpy() for mat in rotation_matrices]))
    
    # Create a SLERP interpolator
    slerp = spt.Slerp(t_original, rots)
    
    # Interpolate at target points using SLERP
    interp_rots = slerp(t_target)
    
    # Convert back to matrices
    interp_matrices = [torch.Tensor(mat) for mat in interp_rots.as_matrix()]
    
    return interp_matrices

def interpolate_rotation_6d(rotation_6d, t_original, t_target):
    """Interpolate 6D rotation.
    
    Args:
        rotation_6d: 6D rotation tensor of shape [T, 6]
        t_original: Original time points
        t_target: Target time points
        target_length: Target sequence length
        
    Returns:
        Interpolated 6D rotation tensor of shape [target_length, 6]
    """
    original_length = rotation_6d.shape[0]
    
    # Convert 6D to rotation matrices
    rotation_matrices = []
    for i in range(original_length):
        rot_mat = rot6d_to_matrix(rotation_6d[i])
        rotation_matrices.append(rot_mat)
    
    # Interpolate rotation matrices
    interp_matrices = interpolate_rotations(rotation_matrices, t_original, t_target)
    
    # Convert back to 6D representation
    rotation_6d_interp = torch.stack([matrix_to_rot6d(matrix) for matrix in interp_matrices])
        
    return rotation_6d_interp

def interp_nd(x_new, x_old, y_old):
    """Interpolate n-dimensional arrays along the first dimension.
    
    Args:
        x_new: New x coordinates, shape [M]
        x_old: Old x coordinates, shape [N]
        y_old: Old y values, shape [N, ...] (can be multi-dimensional)
        
    Returns:
        Interpolated y values, shape [M, ...]
    """
    # Initialize output array with the same shape as y_old except first dimension
    output_shape = (len(x_new),) + y_old.shape[1:]
    y_new = np.zeros(output_shape, dtype=y_old.dtype)
    
    
    # For each element in the output dimensions, perform 1D interpolation (optimzed for 2d)
    for i in range(y_old.shape[1]):
        y_new[:, i] = np.interp(x_new, x_old, y_old[:, i])

    return y_new

def interpolate_gripper(gripper, t_original, t_target, interpolate_gripper_flag=True):
    """Interpolate gripper values.
    
    Args:
        gripper: Gripper tensor of shape [T, 1]
        t_original: Original time points
        t_target: Target time points
        interpolate_gripper_flag: Whether to interpolate gripper values or use zero-order hold
        
    Returns:
        Interpolated gripper tensor of shape [target_length, 1]
    """
    if interpolate_gripper_flag:
        # Normal linear interpolation
        interp_values = interp_nd(t_target, t_original, gripper.numpy())
        return torch.Tensor(interp_values)
    else:
        # Zero-order hold using numpy's searchsorted for efficient index finding
        indices = np.searchsorted(t_original, t_target, side='right') - 1
        indices = np.clip(indices, 0, len(t_original) - 1)  # Ensure indices are in bounds
        gripper_interp = torch.tensor(gripper.numpy()[indices], device=gripper.device)
        
        return gripper_interp

def interpolate_single_arm(action, t_original, t_target, interpolate_gripper_flag=True):
    """Interpolate single arm action.
    
    Args:
        action: Action tensor of shape [T, 10]
        t_original: Original time points
        t_target: Target time points
        target_length: Target sequence length
        interpolate_gripper_flag: Whether to interpolate gripper values
        
    Returns:
        Interpolated action tensor of shape [target_length, 10]
    """
    # Separate components
    pose = action[:, :3]
    rotation_6d = action[:, 3:9]
    gripper = action[:, 9:10]
    
    # Interpolate pose (linear) - replaced np.interp with interp_nd
    pose_interp = torch.Tensor(interp_nd(t_target, t_original, pose.numpy()))
    
    # Interpolate rotation
    rotation_6d_interp = interpolate_rotation_6d(rotation_6d, t_original, t_target)
    
    # Interpolate gripper
    gripper_interp = interpolate_gripper(gripper, t_original, t_target, interpolate_gripper_flag)
    
    # Combine interpolated components
    return torch.cat([pose_interp, rotation_6d_interp, gripper_interp], dim=1)

def interpolate_dual_arm(action, t_original, t_target, target_length, interpolate_gripper_flag=True):
    """Interpolate dual arm action.
    
    Args:
        action: Action tensor of shape [T, 20]
        t_original: Original time points
        t_target: Target time points
        target_length: Target sequence length
        interpolate_gripper_flag: Whether to interpolate gripper values
        
    Returns:
        Interpolated action tensor of shape [target_length, 20]
    """
    # Separate components for arm 1
    pose1 = action[:, :3]
    rotation_6d1 = action[:, 3:9]
    gripper1 = action[:, 9:10]
    
    # Separate components for arm 2
    pose2 = action[:, 10:13]
    rotation_6d2 = action[:, 13:19]
    gripper2 = action[:, 19:20]
    
    # Interpolate poses (linear) - replaced np.interp with interp_nd
    pose1_interp = torch.Tensor(interp_nd(t_target, t_original, pose1.numpy()))
    pose2_interp = torch.Tensor(interp_nd(t_target, t_original, pose2.numpy()))
    
    # Interpolate rotations
    rotation_6d1_interp = interpolate_rotation_6d(rotation_6d1, t_original, t_target)
    rotation_6d2_interp = interpolate_rotation_6d(rotation_6d2, t_original, t_target)
    
    # Interpolate grippers
    gripper1_interp = interpolate_gripper(gripper1, t_original, t_target, interpolate_gripper_flag)
    gripper2_interp = interpolate_gripper(gripper2, t_original, t_target, interpolate_gripper_flag)
    
    # Combine interpolated components
    return torch.cat([
        pose1_interp, rotation_6d1_interp, gripper1_interp,
        pose2_interp, rotation_6d2_interp, gripper2_interp
    ], dim=1)

def interpolate_action(proprio, action, original_hz, target_hz, action_length, interpolate_gripper_flag=True):
    """Interpolate action from original_hz to target_hz and crop to action_length.
    
    Args:
        proprio: Proprioception tensor of shape [1, D]
        action: Action tensor of shape [T, D]
        original_hz: Original frequency
        target_hz: Target frequency
        action_length: Original sequence length
        interpolate_gripper_flag: Whether to interpolate gripper values
        
    Returns:
        Interpolated action tensor of shape [action_length, D]
    """

    if target_hz == original_hz:
        # No interpolation needed
        return action[:action_length]
    
    # Make a copy to avoid modifying the original proprio
    proprio_processed = proprio.clone()
    
    if action.shape[1] == 10:  # Single arm
        proprio_processed[0, 9] = action[0, 9]
    elif action.shape[1] == 20:  # Dual arm
        proprio_processed[0, 9] = action[0, 9]
        proprio_processed[0, 19] = action[0, 19]
    
    # Prepend the processed proprioception as the first action
    action = torch.cat([proprio_processed, action], dim=0)
    # Create time points for original and target sequences
    duration = (action.shape[0] - 1) / original_hz
    t_for_dataset = np.arange(action.shape[0]) / original_hz   # 0, 1/Hz, 2/Hz, ...
    t_sampled     = np.arange(int(duration * target_hz) + 1) / target_hz  # 0, 1/target_hz, ...
    
    # Determine action dimensions based on shape
    if action.shape[1] == 10:  # Single arm
        action_interp = interpolate_single_arm(action, t_for_dataset, t_sampled, interpolate_gripper_flag)
    elif action.shape[1] == 20:  # Dual arm
        action_interp = interpolate_dual_arm(action, t_for_dataset, t_sampled, interpolate_gripper_flag)
    else:
        raise ValueError(f"Unexpected action shape: {action.shape}")
    
    # Crop back to original length if needed
    return action_interp[1:action_length+1]
