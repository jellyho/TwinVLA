import os
import glob
import h5py
import numpy as np
import tensorflow_datasets as tfds
import argparse
from typing import Iterator, Tuple, Any, Type
from conversion_utils import MultiThreadedDatasetBuilder
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for all GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # Handle exception, as memory settings cannot be changed after the program starts
        print(e)

# This function remains the same, as its logic is generic.
def _generate_examples(paths: list) -> Iterator[Tuple[str, Any]]:
    """Yields episodes for a given list of HDF5 files."""
    print(f"[INFO] Generating examples from {len(paths)} paths")
    for path in paths:
        print(f"[INFO] Parsing file: {path}")
        try:
            with h5py.File(path, "r") as f:
                # Check for essential keys
                required_keys = [
                    "/actions/ee_6d_pos",
                    "/actions/ee_quat_pos",
                    "/actions/joint_pos",
                    "/observations/images/back",
                    "/observations/images/wrist_left",
                    "/observations/images/wrist_right",
                    "/observations/states/ee_6d_pos",
                    "/observations/states/ee_quat_pos",
                    "/observations/states/qpos",
                    "/observations/states/qvel",
                    "/observations/states/env_state",
                    "/observations/states/language_instruction"
                ]
                if not all(k in f for k in required_keys):
                    missing_keys = [k for k in required_keys if k not in f]
                    print(f"[WARNING] Missing keys {missing_keys} in {path}, skipping.")
                    continue

                # Load data and ensure consistent lengths
                T = f["/actions/ee_6d_pos"].shape[0]

                head = f["/observations/images/back"].astype(np.uint8)
                left = f["/observations/images/wrist_left"].astype(np.uint8)
                right = f["/observations/images/wrist_right"].astype(np.uint8)

                state_eef_6d_pos = f["/observations/states/ee_6d_pos"].astype(np.float32)
                state_eef_quat_pos = f["/observations/states/ee_quat_pos"].astype(np.float32)
                state_joint_pos = f["/observations/states/joint_pos"].astype(np.float32)

                env_states = f["/observations/states/env_state"].astype(np.float32)

                action_eef_6d_pos = f["/actions/ee_6d_pos"].astype(np.float32)
                action_eef_quat_pos = f["/actions/ee_quat_pos"].astype(np.float32)
                action_joint_pos = f["/actions/joint_pos"].astype(np.float32)

                language_instructions = [
                    s.decode("utf-8") if isinstance(s, bytes) else s
                    for s in f["/observations/states/language_instruction"][()]
                ]

                steps = []
                for i in range(T):
                    step = {
                        "observation": {
                            "image": head[i],
                            "left_wrist_image": left[i],
                            "right_wrist_image": right[i],
                            "joint_pos": state_joint_pos[i],
                            "eef_6d_pos": state_eef_6d_pos[i],
                            "eef_quat_pos": state_eef_quat_pos[i]
                        },
                        "action": {
                            "joint_pos": action_joint_pos[i],
                            "eef_6d_pos": action_eef_6d_pos[i],
                            "eef_quat_pos": action_eef_quat_pos[i]
                        },
                        "discount": np.float32(1.0),
                        "reward": np.float32(1.0 if i == T - 1 else 0.0),
                        "is_first": np.bool_(i == 0),
                        "is_last": np.bool_(i == T - 1),
                        "is_terminal": np.bool_(i == T - 1),
                        "language_instruction": language_instructions[i],
                    }
                    steps.append(step)

                print(f"[INFO] Yielding {len(steps)} steps from {path}")
                yield path, {"steps": steps}
        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {e}")
            continue

def create_tabletop_sim_dataset_builder(task_name: str, data_root: str) -> Type[MultiThreadedDatasetBuilder]:
    """
    Factory function to dynamically create a TFDS Builder class using the type() constructor.
    This is a more robust method than modifying __name__.
    """
    # 1. Dynamically create the class name (same as before)
    class_name = f"{''.join(word.capitalize() for word in task_name.split('_'))}"

    # 2. Define methods to be included in the class as inner functions
    #    This allows the task_name and data_root variables to be used within the functions.
    def _info(self) -> tfds.core.DatasetInfo:
        """Defines the dataset info."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(shape=(480, 640, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "left_wrist_image": tfds.features.Image(shape=(240, 320, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "right_wrist_image": tfds.features.Image(shape=(240, 320, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "joint_pos": tfds.features.Tensor(shape=(14,), dtype=np.float32),
                        "eef_6d_pos": tfds.features.Tensor(shape=(20,), dtype=np.float32),
                        "eef_quat_pos": tfds.features.Tensor(shape=(16,), dtype=np.float32),
                    }),
                    "action": tfds.features.FeaturesDict({
                        "joint_pos": tfds.features.Tensor(shape=(14,), dtype=np.float32),
                        "eef_6d_pos": tfds.features.Tensor(shape=(20,), dtype=np.float32),
                        "eef_quat_pos": tfds.features.Tensor(shape=(16,), dtype=np.float32),
                    }),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
            })
        )

    def _split_paths(self):
        """Returns a dictionary of splits with their corresponding file paths."""
        task_dir = os.path.join(data_root, task_name)
        print(f"[INFO] Searching for data in: {task_dir}")

        train_path = os.path.join(task_dir, "*.hdf5")
        train_files = glob.glob(train_path)

        if not train_files:
            raise FileNotFoundError(f"No .hdf5 files found for task '{task_name}' in {task_dir}")

        print(f"[INFO] Found {len(train_files)} training files from: {train_path}")
        
        if not train_files:
            raise FileNotFoundError(f"No .hdf5 files found for task '{task_name}' in {task_dir}")

        return {
            "train": train_files,
        }

    # 3. Dynamically create the class using type()
    # Format: type(ClassName, (BaseClasses,), {ClassAttributesAndMethods})
    DynamicTabletopBuilder = type(
        class_name,
        (MultiThreadedDatasetBuilder,),
        {
            # Class attributes
            "VERSION": tfds.core.Version("1.0.0"),
            "RELEASE_NOTES": {
                "1.0.0": f"Initial release for the task: {task_name}."
            },
            "N_WORKERS": 10,
            "MAX_PATHS_IN_MEMORY": 100,
            "PARSE_FCN": _generate_examples,
            
            # Class methods
            "_info": _info,
            "_split_paths": _split_paths,
        }
    )

    return DynamicTabletopBuilder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a TFDS dataset for a specific tabletop-sim task.")
    
    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        help="The name of the task directory to process (e.g., 'adjust_bottle')."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="./",
        help="The root directory where the task data (hdf5) folders are stored."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default='./',
        help="The directory to save the generated dataset. Overrides TFDS_DATA_DIR."
    )
    
    args = parser.parse_args()

    # 1. Create the specific builder class for the requested task
    TabletopTaskBuilder = create_tabletop_sim_dataset_builder(task_name=args.task_name, data_root=args.data_root)
    
    # 2. Instantiate the builder
    builder = TabletopTaskBuilder(data_dir=args.output_dir)

    # 3. Run the download and preparation process
    print(f"\n[INFO] Starting dataset build for task: '{args.task_name}'")
    print(f"[INFO] Output directory will be based on the name: '{builder.name}'")
    builder.download_and_prepare()
    print(f"\n[INFO] Dataset build for '{args.task_name}' completed successfully!")