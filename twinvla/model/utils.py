import torch
from transformers import StoppingCriteria
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.init as init

def template_pi0(text, state=None, action=None, stopping_criteria='|'):
    discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized_state))
    if action is None:
        return f'Task: {text}, State: {state_str};\nAction: '
    else:
        return f'Task: {text}, State: {state_str};\nAction: {action}|'


class StopOnChar(StoppingCriteria):
    def __init__(self, tokenizer, stop_char="|"):
        self.tokenizer = tokenizer
        self.stop_char = stop_char

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Get the last generated token for the first sequence in the batch.
        last_token_id = input_ids[0, -1].item()
        # Decode the last token to check if it contains the stop character.
        decoded_token = self.tokenizer.decode([last_token_id])
        if self.stop_char in decoded_token:
            return True
        return False

# Image resizer - PIL Image
def resize_image(image, output_size):
    resized_img = image.resize(output_size, Image.ANTIALIAS)
    return resized_img

def initialize_weights(model):
    """
    Initialize weights for different layers in the model.

    Args:
        model: The model whose weights are to be initialized.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            # He initialization for Conv1d
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)

        elif isinstance(m, nn.Conv2d):
            # He initialization for Conv2d
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.zeros_(m.bias)

        elif isinstance(m, nn.ConvTranspose1d):
            # He initialization for ConvTranspose1d (Transpose Convolution 1D)
            init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # fan_in for transpose
            if m.bias is not None:
                init.zeros_(m.bias)

        elif isinstance(m, nn.ConvTranspose2d):
            # He initialization for ConvTranspose2d (Transpose Convolution 2D)
            init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # fan_in for transpose
            if m.bias is not None:
                init.zeros_(m.bias)

        elif isinstance(m, nn.Linear):
            # Xavier initialization for Linear layers
            init.xavier_normal_(m.weight)
            if m.bias is not None:
                init.zeros_(m.bias)

        elif isinstance(m, nn.BatchNorm1d):
            # Initialize BatchNorm layers
            init.ones_(m.weight)  # Set gamma to 1
            init.zeros_(m.bias)   # Set beta to 0

        elif isinstance(m, nn.BatchNorm2d):
            # Initialize BatchNorm layers for 2D (Conv2d case)
            init.ones_(m.weight)  # Set gamma to 1
            init.zeros_(m.bias)   # Set beta to 0

        elif isinstance(m, nn.LayerNorm):
            # Initialize LayerNorm layer
            if m.weight is not None:  # Check before initialization
                init.ones_(m.weight)  # Set gamma to 1
            if m.bias is not None:
                init.zeros_(m.bias)   # Set beta to 0

        elif isinstance(m, nn.Embedding):
            # Xavier initialization for Embedding layers
            init.xavier_uniform_(m.weight)

        elif isinstance(m, nn.Parameter):
            # Xavier initialization for Embedding layers
            init.normal_(m.weight, std=0.02)

        elif isinstance(m, nn.LSTM):
            # LSTM specific initialization
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'weight_hh' in name:
                    init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'bias' in name:
                    init.zeros_(param)

        elif isinstance(m, nn.GRU):
            # GRU specific initialization
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'weight_hh' in name:
                    init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'bias' in name:
                    init.zeros_(param)

        elif isinstance(m, nn.RNN):
            # RNN specific initialization
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'weight_hh' in name:
                    init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                elif 'bias' in name:
                    init.zeros_(param)
        elif isinstance(m, nn.Parameter):
            # Initialize nn.Parameter if it is directly used
            init.uniform_(m, a=-0.1, b=0.1)  # Uniform initialization

        # Any other layers can be added here with their respective initialization strategies.
