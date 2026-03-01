import math

from twinvla.datasets.datasets import RLDSDataset, RLDSBatchTransform, RLDSBatchIdentity
from twinvla.datasets.rlds.utils.data_utils import PaddedCollatorForActionPrediction
from twinvla.datasets.rlds.oxe.hzs import hz_dict
from twinvla.datasets.rlds.oxe.mixtures import OXE_NAMED_MIXTURES
from torch.utils.data import DataLoader

def load_datasets(model, model_args, training_args, single_arm=True, use_label=True, no_transform=False):
    """
    Load datasets for training/evaluation.
    Supports RLDS datasets and LeRobot datasets.

    Args:
        model: The model instance.
        model_args: Model arguments containing configuration for data processing.
        training_args: Training arguments containing paths and training hyperparameters.
        single_arm: Boolean, whether to load single-arm dataset (SingleVLA) or dual-arm (TwinVLA).
        use_label: Boolean, whether to include labels (unused in current implementation but kept for compatibility).
        no_transform: Boolean, if True, applies identity transform instead of standard preprocessing.

    Returns:
        dataloader: A PyTorch DataLoader yielding batches of data.
        dataset_statistics: A dictionary containing dataset statistics (e.g., mean, std, quantile ranges).
    """
    collator = PaddedCollatorForActionPrediction()

    if '/' in training_args.data_mix:
        # Assume we are using LeRobot dataset
        repo_id = training_args.data_mix
        from twinvla.datasets.lerobot.utils import LeRobotDatasetForTwinVLA, InfiniteShuffleSampler
        dataset = LeRobotDatasetForTwinVLA(repo_id, model.preprocess_inputs)
        dataloader = DataLoader(
            dataset,
            batch_size=training_args.batch_size,
            pin_memory=False,
            num_workers=training_args.num_workers,
            sampler=InfiniteShuffleSampler(dataset),
            collate_fn=collator
        )
        return dataloader, dataset.dataset_statistics

    if model_args.hz_interpolate is not None:
        if training_args.data_mix in OXE_NAMED_MIXTURES:
            dataset_names = [x[0] for x in OXE_NAMED_MIXTURES[training_args.data_mix]]
        else:
            dataset_names = [training_args.data_mix]
        max_hz = max([hz_dict[name] for name in dataset_names])
        future_action_window_size = math.ceil(model_args.action_len * (max_hz / model_args.hz_interpolate)) - 1
    else:
        future_action_window_size = model_args.action_len - 1

    if no_transform:
        batch_transform = RLDSBatchIdentity(
            single_arm=single_arm,
            hz_interpolate=model_args.hz_interpolate,
            interpolate_gripper=model_args.interpolate_gripper,
            action_len=model_args.action_len,
        )
    else:
        if single_arm:
            batch_transform = RLDSBatchTransform(
                process_inputs_fn=model.preprocess_inputs,
                window_size=1,
                single_arm=single_arm,
                chunk_hz=model.config.action_head=='FAST',
                hz_interpolate=model_args.hz_interpolate,
                interpolate_gripper=model_args.interpolate_gripper,
                action_len=model_args.action_len,
                knowledge_insulation=model.config.knowledge_insulation
            )
        else:
            batch_transform = RLDSBatchTransform(
                process_inputs_fn=model.preprocess_inputs,
                window_size=1,
                single_arm=single_arm,
                chunk_hz=model.config.action_head=='FAST',
                hz_interpolate=model_args.hz_interpolate,
                interpolate_gripper=model_args.interpolate_gripper,
                action_len=model_args.action_len
            )

    if single_arm:
        dataset = RLDSDataset(
            batch_size=training_args.batch_size,
            collate_fn=collator,
            data_root_dir=training_args.data_root_dir,
            data_mix=training_args.data_mix,
            batch_transform=batch_transform,
            shuffle_buffer_size=training_args.shuffle_buffer_size,
            train=True,
            window_size=1,
            future_action_window_size=future_action_window_size,
            enable_autotune=training_args.enable_autotune,
            num_parallel_calls=training_args.num_parallel_calls,
            quantile_norm=model_args.normalization == 'quantile',
            image_aug=training_args.image_aug,
            global_normalization=model_args.global_normalization,
            dataset_statistics_path=None,
            single_arm=single_arm,
            num_workers=training_args.num_workers
        )
    else:
        dataset = RLDSDataset(
            batch_size=training_args.batch_size,
            collate_fn=collator,
            data_root_dir=training_args.data_root_dir,
            data_mix=training_args.data_mix,
            batch_transform=batch_transform,
            shuffle_buffer_size=training_args.shuffle_buffer_size,
            train=True,
            window_size=1,
            future_action_window_size=future_action_window_size,
            enable_autotune=training_args.enable_autotune,
            num_parallel_calls=training_args.num_parallel_calls,
            quantile_norm=model.config.normalization == 'quantile',
            image_aug=training_args.image_aug,
            global_normalization=model.config.global_normalization,
            dataset_statistics_path=model_args.singlevla_pretrained_path if model_args.singlevla_pretrained_path else training_args.pretrained_path,
            single_arm=single_arm,
            num_workers=training_args.num_workers
        )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        sampler=None,
        collate_fn=lambda x: x[0],
        pin_memory=True,
        num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
    )

    return dataloader, dataset.dataset_statistics