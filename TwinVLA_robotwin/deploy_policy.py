# import packages and module here
import numpy as np
from scipy.spatial.transform import Rotation as R
from twinvla.model.twinvla import TwinVLA

def quaternion_to_6d(quat):
    # using scipy Rotation
    r = R.from_quat(quat, scalar_first=True)
    rot_mat = r.as_matrix()
    rot1 = rot_mat[:, 0]
    rot2 = rot_mat[:, 1]
    return np.concatenate([rot1, rot2], axis=-1)

def sixd_to_quaternion(rotation_6d):
    """Convert a 6D rotation representation to a quaternion.
    
    Args:
        rotation_6d: A 6D rotation representation (first two columns of the rotation matrix)
        
    Returns:
        A 3x3 rotation matrix
    """
    # Extract the first two columns
    col0 = rotation_6d[:3]
    col1 = rotation_6d[3:6]
    eps = 1e-6
    # ── Gram–Schmidt orthonormalization ─────────────────────────────
    x = col0 / (np.linalg.norm(col0) + eps)

    proj   = np.dot(x, col1)
    y_raw  = col1 - proj * x
    y_norm = np.linalg.norm(y_raw) + eps
    y      = y_raw / y_norm

    z = np.cross(x, y)
    z = z / (np.linalg.norm(z) + eps)

    # Assemble columns into a rotation matrix
    Rm = np.stack([x, y, z], axis=1)   # shape (3,3)

    r = R.from_matrix(Rm)
    quat = r.as_quat()
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])  # convert to (w, x, y, z)
    return quat


def generate_proprioception(observation):
    # observation["endpose"]["left_gripper"]
    left_arm_quat = np.array(observation["endpose"]["left_endpose"])
    right_arm_quat = np.array(observation["endpose"]["right_endpose"])
    
    left_arm_6d = np.concatenate([
        left_arm_quat[0:3],
        quaternion_to_6d(left_arm_quat[3:7]),
        np.array([observation["endpose"]["left_gripper"]])
        ], axis=0)
    right_arm_6d = np.concatenate([
        right_arm_quat[0:3],
        quaternion_to_6d(right_arm_quat[3:7]),
        np.array([observation["endpose"]["right_gripper"]])
        ], axis=0)
    proprio = np.concatenate([left_arm_6d, right_arm_6d], axis=0)
    return proprio

def convert_to_quat_action(actions: np.array) -> np.array:
    """
    original actions: [seq_num, 20] (20: left_xyz(3) + left_6d(6) + left_gripper(1) + right_xyz(3) + right_6d(6) + right_gripper(1))
    converted actions: [seq_num, 18] (18: left_xyz(3) + left_quat(4) + left_gripper(1) + right_xyz(3) + right_quat(4) + right_gripper(1))
    """
    converted_actions = []
    for action in actions:
        left_xyz = action[0:3]
        left_6d = action[3:9]
        left_gripper = action[9:10]
        right_xyz = action[10:13]
        right_6d = action[13:19]
        right_gripper = action[19:20]

        left_quat = sixd_to_quaternion(left_6d)
        right_quat = sixd_to_quaternion(right_6d)

        converted_action = np.concatenate([left_xyz, left_quat, left_gripper, right_xyz, right_quat, right_gripper], axis=0)
        converted_actions.append(converted_action)
    return np.array(converted_actions)

def encode_obs(observation, instruction):
    proprio = generate_proprioception(observation)
    obs = dict(
        proprio=proprio,
        image=observation["observation"]["head_camera"]["rgb"],
        image_wrist_l=observation["observation"]["left_camera"]["rgb"],
        image_wrist_r=observation["observation"]["right_camera"]["rgb"],
        instruction=instruction,
    )
    return obs

def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    model = TwinVLA(pretrained_path=usr_args['saved_model_path'], device='cuda')
    return model  # return your policy model


def eval(TASK_ENV, model, observation):
    """
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    obs = encode_obs(observation, TASK_ENV.get_instruction()) 

    actions = model.predict_action(f'robotwin_{TASK_ENV.task_name}', **obs)
    actions = convert_to_quat_action(actions)

    for action in actions:  # Execute each step of the action
        # see for https://robotwin-platform.github.io/doc/control-robot.md more details
        # TASK_ENV.take_action(action, action_type='qpos') # joint control: [left_arm_joints + left_gripper + right_arm_joints + right_gripper]
        TASK_ENV.take_action(action, action_type='ee') # endpose control: [left_end_effector_pose (xyz + quaternion) + left_gripper + right_end_effector_pose + right_gripper]
        # TASK_ENV.take_action(action, action_type='delta_ee') # delta endpose control: [left_end_effector_delta (xyz + quaternion) + left_gripper + right_end_effector_delta + right_gripper]
        observation = TASK_ENV.get_obs()
        # obs = encode_obs(observation, TASK_ENV.get_instruction())


def reset_model(model):  
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    pass
