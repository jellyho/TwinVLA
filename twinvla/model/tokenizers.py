from typing import List, Union
import numpy as np
import scipy.stats as stats
from transformers import AutoProcessor, PreTrainedTokenizerBase
    
class FASTTokenizer:
    def __init__(
        self, tokenizer, action_dim=32, vocab_start=None
    ) -> None:
        # Instantiate FAST tokenizer
        self.fast_tokenizer = AutoProcessor.from_pretrained("physical-intelligence/fast", trust_remote_code=True, local_files_only=False)
        self.tokenizer = tokenizer
        if vocab_start is None:
            self.vocab_start = len(self.tokenizer) - 1000
        else:
            self.vocab_start = vocab_start
        print(f'FAST vocab starts from {self.vocab_start}')

    def __call__(self, action):
        action_tokens = self.fast_tokenizer(action[None])[0]
        action_tokens = self.vocab_start - np.array(action_tokens) ## HARD CODED
        return list(action_tokens)

    def decode(self, action_ids, action_len, action_dim):
        action_tokens = self.vocab_start - np.array(action_ids)
        action = self.fast_tokenizer.decode([action_tokens[0]], time_horizon=action_len, action_dim=action_dim)
        return action

class StateTokenizer:
    def __init__(self, tokenizer, return_text=False):
        self.tokenizer = tokenizer

    def __call__(self, state):
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
        state_str = " ".join(map(str, discretized_state))
        state_ids = self.tokenizer(state_str).input_ids[1:-1]
        return state_ids

class ActionTokenizer:
    def __init__(
        self,
        tokenizer,
        bins: int = 256,
        min_action: int = -1,
        max_action: int = 1,
        add_action_end_flag=False,
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        """

        self.tokenizer, self.n_bins, self.min_action, self.max_action = (
            tokenizer,
            bins,
            min_action,
            max_action,
        )

        special_tokens_count = sum(
            len(tokens) if isinstance(tokens, list) else 1
            for key, tokens in tokenizer.special_tokens_map_extended.items()
        )
        self.special_tokens_count = special_tokens_count
        offset = 10
        self.tokenizer_orig_size = (
            self.tokenizer.vocab_size - self.special_tokens_count - offset
        )

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(
            self.tokenizer_orig_size - (self.n_bins + 1)
        )
        self.action_token_end_idx = self.tokenizer_orig_size

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        action = np.clip(
            action, a_min=float(self.min_action), a_max=float(self.max_action)
        )
        discretized_action = np.digitize(action, self.bins)

        # Handle single element vs. batch
        if len(discretized_action.shape) == 1:
            return self.tokenizer.decode(
                list(self.tokenizer_orig_size - discretized_action)
            )
        else:
            return self.tokenizer.batch_decode(
                (self.tokenizer_orig_size - discretized_action).tolist()
            )

    def encode_actions_to_token_ids(self, action: np.ndarray) -> np.ndarray:
        action = np.clip(
            action, a_min=float(self.min_action), a_max=float(self.max_action)
        )
        discretized_action = np.digitize(action, self.bins)

        if len(discretized_action.shape) == 1:
            return list(self.tokenizer_orig_size - discretized_action)
        else:
            return np.array(self.tokenizer_orig_size - discretized_action).tolist()

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        discretized_actions = self.tokenizer_orig_size - action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )

        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins