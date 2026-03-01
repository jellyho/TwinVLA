import os
import glob
import h5py
import numpy as np
import tensorflow_datasets as tfds
import argparse
from typing import Iterator, Tuple, Any, Type
from conversion_utils import MultiThreadedDatasetBuilder
from conversion_utils import quat_to_6d, sixd_to_quat
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
                    "/action",
                    "/observation/camera_center_image",
                    "/observation/camera_left_image",
                    "/observation/camera_right_image",
                    "/observation/eef_6d_pose"
                ]
                if not all(k in f for k in required_keys):
                    missing_keys = [k for k in required_keys if k not in f]
                    print(f"[WARNING] Missing keys {missing_keys} in {path}, skipping.")
                    continue

                # Load data and ensure consistent lengths
                T = f["/action"].shape[0]
                if T <= 1:
                    print(f"[WARNING] Episode in {path} has length {T}, skipping.")
                    continue

                actions = f["/action"].astype(np.float32)
                head = f["/observation/camera_center_image"].astype(np.uint8)
                left = f["/observation/camera_left_image"].astype(np.uint8)
                right = f["/observation/camera_right_image"].astype(np.uint8)
                eef_states = f["/observation/eef_6d_pose"].astype(np.float32)

                if 'cabbage' in path:
                    print('cabbage')
                    instruction = ['Pick up the cabbage and place into the pot']
                elif 'carrot' in path:
                    print('carrot')
                    instruction = ['Pick up the carrot and place into the pot']
                elif 'eggplant' in path:
                    print('eggplant')
                    instruction = ['Pick up the eggplant and place into the pot']
                else:
                    instruction = ['Fold the towel then rotate it and fold again']
                    instruction = ['pull the wrench and place it on the bowl']
                # T -= 1
                
                steps = []
                for i in range(T):
                    quat_actions = actions[i]
                    left_xyz, left_quat, left_grpip = quat_actions[0:3], quat_actions[3:7], quat_actions[7:8]
                    right_xyz, right_quat, right_grpip = quat_actions[8:11], quat_actions[11:15], quat_actions[15:16]
                    actions_6d = np.concatenate([
                        left_xyz,
                        quat_to_6d(left_quat, scalar_first=False),
                        left_grpip,
                        right_xyz,
                        quat_to_6d(right_quat, scalar_first=False),
                        right_grpip
                    ], axis=0)
                    step = {
                        "observation": {
                            "image": head[i],
                            "left_wrist_image": left[i],
                            "right_wrist_image": right[i],
                            "eef_state": eef_states[i]
                        },
                        "action": actions_6d.astype(np.float32),
                        "discount": np.float32(1.0),
                        "reward": np.float32(1.0 if i == T - 1 else 0.0),
                        "is_first": np.bool_(i == 0),
                        "is_last": np.bool_(i == T - 1),
                        "is_terminal": np.bool_(i == T - 1),
                        "language_instruction": instruction,
                    }
                    steps.append(step)

                print(f"[INFO] Yielding {len(steps)} steps from {path}")
                yield path, {"steps": steps, "episode_metadata": {"file_path": path}}
        except Exception as e:
            print(f"[ERROR] Failed to process {path}: {e}")
            continue

def create_aloha_dataset_builder(task_name: str, data_root: str) -> Type[MultiThreadedDatasetBuilder]:
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
                        "image": tfds.features.Image(shape=(240, 320, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "left_wrist_image": tfds.features.Image(shape=(240, 320, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "right_wrist_image": tfds.features.Image(shape=(240, 320, 3), dtype=np.uint8, encoding_format="jpeg"),
                        "eef_state": tfds.features.Tensor(shape=(20,), dtype=np.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(20,), dtype=np.float32),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Sequence(tfds.features.Text()),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                }),
            })
        )

    def _split_paths(self):
        """Returns a dictionary of splits with their corresponding file paths."""
        train_path = os.path.join(data_root, task_name, "*.hdf5")
        train_files = glob.glob(train_path)

        print(f"[INFO] Found {len(train_files)} training files from: {train_path}")
        
        if not train_files:
            raise FileNotFoundError(f"No .hdf5 files found for task '{task_name}' in {data_root}")

        return {
            "train": train_files,
        }

    # 3. Dynamically create the class using type()
    # Format: type(ClassName, (BaseClasses,), {ClassAttributesAndMethods})
    DynamicAlohaBuilder = type(
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

    return DynamicAlohaBuilder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a TFDS dataset for a specific Anubis task.")
    
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
    AlohaTaskBuilder = create_aloha_dataset_builder(task_name=args.task_name, data_root=args.data_root)
    
    # 2. Instantiate the builder
    builder = AlohaTaskBuilder(data_dir=args.output_dir)

    # 3. Run the download and preparation process
    print(f"\n[INFO] Starting dataset build for task: '{args.task_name}'")
    print(f"[INFO] Output directory will be based on the name: '{builder.name}'")
    builder.download_and_prepare()
    print(f"\n[INFO] Dataset build for '{args.task_name}' completed successfully!")