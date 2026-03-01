from twinvla.model.utils import StopOnChar, initialize_weights
from twinvla.model.tokenizers import FASTTokenizer
from twinvla.model.heads.action_heads import MLPHead, DiTPolicy
from twinvla.model.heads.utils import MultiheadAttentionPooling, AveragePooling
from transformers import StoppingCriteriaList
from transformers import (
    PretrainedConfig,
)

from transformers.modeling_outputs import CausalLMOutputWithPast

from transformers import (
    AutoConfig,
    AutoModel,
)
import torch
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import time
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel


# Architecture : method
modeling_head_type = {
    'FAST':'tokenization',
    'DP':'denoising',
    'ACT':'regression',
    'DiT':'denoising',
    'XDiT':'denoising'
}

class SingleVLAConfig(PretrainedConfig):
    model_type = "SingleVLAForConditionalGeneration"
    
    def __init__(self, **kwargs):
        # General
        self.action_dim = kwargs.pop("action_dim", None)
        self.action_len = kwargs.pop("action_len", None)
        self.action_head = kwargs.pop("action_head", None)
        self.action_head_hidden_dim = kwargs.pop("action_head_hidden_dim", None)
        self.state_dim = kwargs.pop("state_dim", None)
        self.modeling = None

        # Dataset
        self.normalization = kwargs.pop("normalization", None)
        self.global_normalization = kwargs.pop("global_normalization", None)
        self.image_size = kwargs.pop("image_size", None)
        
        # For tokenization
        self.vocab_start = kwargs.pop("vocab_start", None)
        self.return_text = kwargs.pop("return_text", None)
        self.stopping_token = kwargs.pop("stopping_token", '|')

        # For denoising
        self.denoiser = kwargs.pop("DDIM", None)
        self.train_denoising_steps = kwargs.pop("train_denoising_steps", None)
        self.test_denoising_steps = kwargs.pop("test_denoising_steps", None)
        self.diffusion_batch = kwargs.pop("diffusion_batch", 32)
        self.num_readouts = kwargs.pop("num_readouts", None)
        self.readout_token_as_eos = kwargs.pop("readout_token_as_eos", False)
        self.aggregation = kwargs.pop("aggregation", None)
        self.dit_size = kwargs.pop("dit_size", 'DiT-L')
        self.enable_cfg = kwargs.pop('enable_cfg', True)
        self.knowledge_insulation = kwargs.pop('knowledge_insulation', False)
        super().__init__(**kwargs)

# Implementable functions for SingleVLA
class SingleVLAMetaModel:
    def init_model(self, config, **kwargs):
        self.config = config
        ## Initialize modules
        self.init_processor_tokenizer(self.config)
        self.init_state_embeds(self.config)
        self.init_action_head(self.config, tokenizer=self.tokenizer)
        self.prepare_inputs_ids()

    def init_processor_tokenizer(self, config):
        raise NotImplementedError  

    def text_backbone(self):
        raise NotImplementedError 

    def hidden_dim(self):
        raise NotImplementedError  

    def vision_backbone(self):
        raise NotImplementedError   

    def image_seq_len(self):
        raise NotImplementedError

    def image_start_token(self):
        return None

    def image_end_token(self):
        return None

    def process_image(self, image):
        raise NotImplementedError 

    def image_embeds(self, pixel_values):
        raise NotImplementedError

    def system_prompt(self):
        return ''

    ##############################################
    def language_embeds(self, input_ids):
        return self.get_input_embeddings()(input_ids)
        
    def image_ids(self):
        start = self.image_start_token()
        end = self.image_end_token()
        if start is not None:
            dummy = [start] + [0] * self.image_seq_len()
        else:
            dummy = [0] * self.image_seq_len()
        if end is not None:
            dummy = dummy + [end]
        return dummy
    
    def image_modal_ids(self, modal_id):
        start = self.image_start_token()
        end = self.image_end_token()
        if start is not None:
            dummy = [0] + [modal_id] * self.image_seq_len()
        else:
            dummy = [modal_id] * self.image_seq_len()
        if end is not None:
            dummy = dummy + [0]
        return dummy
    
    def base_prompt(self, task):
        return self.system_prompt()+f"""You are a robotic decision-making assistant.  
            Given a task and robotic system states (Base, Arm), determine the appropriate action.
            What action should the robot take to accomplish {task}?
            Base State: 
        """

    def prepare_inputs_ids(self):
        self.ln_id = self.tokenizer("\n", add_special_tokens=False)['input_ids']
        self.sep_arm_id = self.tokenizer("Arm State: ", add_special_tokens=False)['input_ids']
        self.sep_action_id = self.tokenizer("Action: ", add_special_tokens=False)['input_ids']
        self.sep_stopping_id = self.tokenizer(self.config.stopping_token, add_special_tokens=False)['input_ids']
        self.image_id = self.image_ids()

    def init_state_embeds(self, config):
        state_dim = config.state_dim
        self.state_dim = state_dim
        hidden_size = self.hidden_dim()

        self.embed_arm_state = torch.nn.Linear(state_dim, hidden_size)
        self.embed_arm_state.weight.data.normal_(mean=0.0, std=0.02)
        self.embed_arm_state.bias.data.zero_()
        ##
    def init_action_head(self, config, tokenizer=None):
        if config.knowledge_insulation:
            self.action_tokenizer = FASTTokenizer(tokenizer)
        with torch.no_grad():
            if config.num_readouts != 0 and not config.readout_token_as_eos:       
                self.action_token = nn.Parameter(torch.zeros(config.num_readouts, self.hidden_dim()))
                self.action_token.data = self.get_input_embeddings().weight.mean(dim=0).unsqueeze(0).repeat(config.num_readouts, 1)
                self.action_token.requires_grad_(True)
            if config.action_head == 'DiT':
                dit_size = getattr(config, 'dit_size', 'DiT-L')
                self.action_head = DiTPolicy(
                    model_type=dit_size, 
                    token_size=self.hidden_dim(), 
                    in_channels=config.action_dim, 
                    state_dim=config.state_dim,
                    future_action_window_size=config.action_len,
                    hidden_dim=config.action_head_hidden_dim,
                    diffusion_steps=config.train_denoising_steps,
                    test_denoising_steps=config.test_denoising_steps,
                    denoiser=config.denoiser,
                    enable_cfg=config.enable_cfg
                )
            self.action_head.to(dtype=torch.float32) # Test whether highier precision only for action helps
            self.aggregation = getattr(config, 'aggregation', None)
            if self.aggregation is not None and config.num_readouts != 1:
                if self.aggregation == 'average':
                    self.agg = AveragePooling(self.hidden_dim())
                elif self.aggregation == 'attention':
                    self.agg = MultiheadAttentionPooling(self.hidden_dim(), 8)
                else:
                    self.agg = nn.Identity()
            else:
                self.agg = nn.Identity()

    ##
    def preprocess_inputs(self, image, image_wrist, instruction, action=None):
        # Input: |Sys|"Input: \nWhat~{task}?\nGiven: \nBase: "|<primary_image>|"\n"|"Arm: "|<wrist_image>|<proprio>|"\n\n"|Action: "|FAST|Action-Tokens|
        # Co/In: |ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc|rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr|
        # Modal: |00000000000000000000000000000000000000000000|111111111111111|0000|0000000|2222222222222|    4    |000000|000000000|6666|5555555555555|
        # Label: |-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100  |0000|0000000000000|
        ## Common tokens
        prefix_ids = self.tokenizer(self.base_prompt(instruction), add_special_tokens=False)['input_ids']
        common_input_ids = prefix_ids + self.image_id + self.ln_id
        common_modal_ids = [0] * len(prefix_ids) + self.image_modal_ids(1) + [0] * len(self.ln_id)

        ## Seperate tokens
        sep_input_ids_R = self.sep_arm_id + self.image_id + [0] + self.ln_id + self.ln_id + self.sep_action_id
        sep_modal_ids_R = [0] * len(self.sep_arm_id) + self.image_modal_ids(2) + [4] + (2 * len(self.ln_id) + len(self.sep_action_id)) * [0]

        # Action tokens
        readouts = self.config.num_readouts
        if readouts == 0: # We are going to aggregate all the outputs
            label_length = 0
        else:
            label_length = 0
            if action is not None and self.config.knowledge_insulation:
                action_tokens = self.action_tokenizer(action)
                sep_input_ids_R += action_tokens
                sep_modal_ids_R += [6] * len(action_tokens)
                label_length += len(action_tokens)
            sep_input_ids_R += [0] * readouts
            sep_modal_ids_R += [5] * readouts

        # torch Tensor
        ci_ids = torch.tensor([0] * len(common_input_ids) + [1] * len(sep_input_ids_R))
        modal_ids = torch.tensor(common_modal_ids + sep_modal_ids_R)
        input_ids = torch.tensor(common_input_ids + sep_input_ids_R)
        label_ids = torch.tensor([0] * (len(input_ids) - label_length - readouts) + [1] * label_length + [0] * readouts)
        attention_mask = torch.tensor([1] * len(input_ids))

        # image preprocessing
        pixel_values_primary = self.process_image(image)
        pixel_values_wrist = self.process_image(image_wrist)

        assert len(ci_ids) == len(modal_ids) == len(input_ids) == len(label_ids), f"{len(ci_ids)},{len(modal_ids)},{len(input_ids)},{len(label_ids)}, {ci_ids},{modal_ids},{input_ids},{label_ids}"
        
        return dict(
            ci_ids=ci_ids,
            modal_ids=modal_ids,
            input_ids=input_ids,
            label_ids=label_ids,
            pixel_values_primary=pixel_values_primary,
            pixel_values_wrist=pixel_values_wrist,
            attention_mask=attention_mask
        )

    ##
    def prepare_embeds(self, batch):
        image_embeds_primary = self.image_embeds(batch['pixel_values_primary'])
        image_embeds_wrist = self.image_embeds(batch['pixel_values_wrist'])
        input_embeds = self.language_embeds(batch['input_ids']) 
        state_embeds = self.encode_state(batch['proprio'])
        ## Insert modal_embeds
        input_embeds = self.insert_embeds(input_embeds, image_embeds_primary, batch['modal_ids'], 1)
        input_embeds = self.insert_embeds(input_embeds, image_embeds_wrist, batch['modal_ids'], 2)
        input_embeds = self.insert_embeds(input_embeds, state_embeds, batch['modal_ids'], 4)

        # Insert action_embeds if modeling is regression
        if self.config.num_readouts != 0 and not self.config.readout_token_as_eos:
            action_embeds = self.action_token.repeat(input_embeds.shape[0], 1)
            input_embeds = self.insert_embeds(input_embeds, action_embeds, batch['modal_ids'], 5)

        return input_embeds.contiguous(), batch['attention_mask']

    def insert_embeds(self, input_embeds, modal_embeds, modal_ids, target_modal_id):
        modal_mask = modal_ids == target_modal_id
        input_embeds[modal_mask] = modal_embeds.reshape(-1, input_embeds.shape[-1]).to(dtype=input_embeds.dtype)
        return input_embeds

    def encode_state(self, state):
        # b, l, dim
        arm_state_embeddings = self.embed_arm_state(state)
        return arm_state_embeddings

    ##
    def forward_action_head(self, outputs, batch):
        output = {}
        output['loss'] = 0
        if self.config.knowledge_insulation:
            logits = outputs.logits
            labels = batch['labels']
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            token_loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )
            output['token_loss'] = token_loss.item()
            output['loss'] += token_loss
            output['logits'] = logits

            hidden_state = outputs.hidden_states.detach()
        else:
            hidden_state = outputs.hidden_states[-1]

        bsz = batch['modal_ids'].shape[0]
        if self.config.num_readouts == 0:
            action_token_label = batch['attention_mask']
        else:
            action_token_label = batch['modal_ids'] == 5
        action_token = hidden_state[action_token_label].reshape(bsz, -1, self.hidden_dim())
        ## agg
        action_token = self.agg(action_token)
        loss = self.action_head(batch['action'], action_token, batch['proprio'][:, 0, :]) # B, 1, D
        output['loss'] += loss
        output['denoising_loss'] = loss.item()

        return output

    def inference_action_head(self, logits=None, hidden_states=None, states=None, action_ids=None, action_len=None, action_dim=None, cfg=1.0, num_denoising_steps=10):
        action_head_dtype = self.action_head.dtype
        normalized_action = self.action_head.denoise(hidden_states.to(dtype=action_head_dtype), states.to(dtype=action_head_dtype), cfg=cfg, denoising_steps=num_denoising_steps)
        return normalized_action
    
    def create_4d_knowledge_insulation_attention_mask(self, attn_mask_1d, ci_ids):
        device = attn_mask_1d.device
        B, seqlen = attn_mask_1d.shape

        base_causal_mask = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool, device=device))
        causal_mask = base_causal_mask.unsqueeze(0).expand(B, -1, -1).clone()  # (B, seqlen, seqlen)

        ci_key = ci_ids.unsqueeze(1)
        ci_query = ci_ids.unsqueeze(2)

        readout_mask = (ci_query == 5) & (ci_key == 6)
        causal_mask[readout_mask] = 0

        valid_mask = attn_mask_1d.bool().unsqueeze(1)      # (B, 1, seqlen)
        valid_mask = valid_mask.expand(-1, seqlen, -1) 
        causal_mask = causal_mask & valid_mask
        causal_mask = causal_mask.unsqueeze(1)

        ki_mask = torch.full((B, 1, seqlen, seqlen), torch.finfo(self.dtype).min, device=device)
        ki_mask.masked_fill_(causal_mask, 0.0)
        return ki_mask
    
    ## Assuming using Qwen2
    def knowledge_insulation_forward(self, input_embeds, batch):
        batch = batch.to(device=self.device, dtype=torch.bfloat16)
        ci_ids = batch['ci_ids']
        inputs_embeds, attention_mask = self.prepare_embeds(batch)
        input_embeds = input_embeds

        past_seen_tokens = 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        position_ids = cache_position.unsqueeze(0)

        hidden_states = input_embeds

        position_embeddings = self.text_backbone().model.rotary_emb(hidden_states, position_ids)
        attention_mask = self.create_4d_knowledge_insulation_attention_mask(attention_mask, ci_ids)

        for decoder_layer in self.text_backbone().model.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = layer_outputs[0].contiguous()

        hidden_states = self.text_backbone().model.norm(hidden_states)

        logits = self.text_backbone().lm_head(hidden_states)

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states,
            attentions=None
        )

    
    # Calulate loss
    def base_forward(self, batch, until_action_head=True):
        batch = batch.to(device=self.device, dtype=self.model.dtype)
        inputs_embeds, attention_mask = self.prepare_embeds(batch)
        if self.config.knowledge_insulation:
            outputs = self.knowledge_insulation_forward(inputs_embeds, batch)
        else:
            inputs_embeds, attention_mask = self.prepare_embeds(batch)
            outputs = self.text_backbone()(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        if until_action_head:
            outputs = self.forward_action_head(outputs, batch)
        return outputs
    
    def inference(self, batch, max_new_tokens=128, action_len=10, action_dim=7, cfg=1.0, num_denoising_steps=10, return_action_token=False):
        inputs_embeds, attention_mask = self.prepare_embeds(batch)
        outputs = self.text_backbone()(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        bsz = batch['modal_ids'].shape[0]
        hidden_state = outputs.hidden_states[-1]
        if self.config.num_readouts == 0:
            action_token_label = batch['attention_mask']
        else:
            action_token_label = batch['modal_ids'] == 5
        action_token = hidden_state[action_token_label].reshape(bsz, -1, self.hidden_dim())
        action_head_dtype = self.action_head.dtype

        # agg
        action_token = self.agg(action_token)

        normalized_action = self.inference_action_head(
            hidden_states=action_token.to(dtype=action_head_dtype), 
            states=batch['proprio'][:, 0, :].to(dtype=action_head_dtype), 
            action_len=action_len,
            num_denoising_steps=num_denoising_steps
        )
        normalized_action = normalized_action.reshape(-1, action_len, action_dim)

        if return_action_token:
            return normalized_action, action_token
        return normalized_action

# base_models.py
class TwinVLAConfig(PretrainedConfig):
    model_type = "TwinVLAForConditionalGeneration"
    
    def __init__(self, **kwargs):
        # General
        self.singlevla_config = kwargs.pop("singlevla_config", None)

        # Inherits form SingleVLA
        self.action_dim = kwargs.pop("action_dim", None)
        self.action_len = kwargs.pop("action_len", None)
        self.action_head = kwargs.pop("action_head", None)
        self.state_dim = kwargs.pop("state_dim", None)
        self.normalization = kwargs.pop("normalization", None)
        self.global_normalization = kwargs.pop("global_normalization", None)
        self.modeling = kwargs.pop("modeling", None)
        self.action_head = kwargs.pop("action_head", None)
        self.denoiser = kwargs.pop("denoiser", None)

        # For TwinVLA
        self.share_vision = kwargs.pop("share_vision", True)
        self.share_decoder = kwargs.pop("share_decoder", True)
        self.share_embed_tokens = kwargs.pop("share_embed_tokens", True)
        self.attn_reweighting = kwargs.pop("attn_reweighting", True)
        self.dit_scratch = kwargs.pop("dit_scratch", False)
        self.enable_moe = kwargs.pop("enable_moe", True)
        self.enable_joint_attn = kwargs.pop("enable_joint_attn", True)
        super().__init__(**kwargs)

import copy

class TwinVLAMetaModel:
    def copy_backbone_target(self):
        return 'langauge_model.model'
    
    def copy_vision_target(self):
        return 'vision_model'

    def copy_embed_tokens_target(self):
        return 'language_model.model.embed_tokens'
    
    def copy_norm_target(self):
        return 'language_model.model.norm'
    
    def copy_pos_embed_target(self):
        return 'language_model.model.rotary_emb'
    
    def copy_other_target(self):
        ## This is for the case that we need to copy other modules
        return None
    
    def init_processor_tokenizer(self, config):
        raise NotImplementedError  
    
    def process_image(self, image):
        raise NotImplementedError

    def image_embeds(self, pixel_values):
        raise NotImplementedError

    def in_layernorm(self, model, idx):
        raise NotImplementedError
    
    def out_layernorm(self, model, idx):
        raise NotImplementedError
    
    def attn(self, model, idx):
        raise NotImplementedError
    
    def mlp(self, model, idx):
        raise NotImplementedError

    ##############################################
    def text_backbone(self):
        pass

    def vision_backbone(self):
        pass

    def image_ids(self):
        start = self.image_start_token()
        end = self.image_end_token()
        if start is not None:
            dummy = [start] + [0] * self.image_seq_len()
        else:
            dummy = [0] * self.image_seq_len()
        if end is not None:
            dummy = dummy + [end]
        return dummy
    
    def image_modal_ids(self, modal_id):
        start = self.image_start_token()
        end = self.image_end_token()
        if start is not None:
            dummy = [0] + [modal_id] * self.image_seq_len()
        else:
            dummy = [modal_id] * self.image_seq_len()
        if end is not None:
            dummy = dummy + [0]
        return dummy
    
    def system_prompt(self):
        return ''
    
    def base_prompt(self, task):
        return self.system_prompt()+f"""You are a robotic decision-making assistant.  
            Given a task and robotic system states (Base, Arm), determine the appropriate action.
            What action should the robot take to accomplish {task}?
            Base State: 
        """
    
    ## Assume we are only using common embed tokens for now
    def language_embeds(self, input_ids, right=True):
        if self.config.share_embed_tokens:
            return self.embed_tokens_c(input_ids)
        else:
            if right:
                return self.embed_tokens_r(input_ids)
            else:
                return self.embed_tokens_l(input_ids)
    
    ###
    def prepare_inputs_ids(self):
        self.ln_id = self.tokenizer("\n", add_special_tokens=False)['input_ids']
        self.sep_arm_id = self.tokenizer("Arm State: ", add_special_tokens=False)['input_ids']
        self.sep_action_id = self.tokenizer("Action: ", add_special_tokens=False)['input_ids']
        self.image_id = self.image_ids()

    ###
    def preprocess_inputs(self, image, image_wrist_r, image_wrist_l, instruction, action=None):
        # Input: |Sys|"Input: \nWhat~{task}?\nGiven: \nBase: "|<primary_image>|"\n"|"Arm: "|<wrist_image>|<proprio>|"\n\n"|Action: "|Action-Tokens|"Arm: "|<wrist_image>|<proprio>|"\n\n"|Action: "|Action-Tokens|
        # Co/In: |ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc|rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr|rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr|
        # Modal: |00000000000000000000000000000000000000000000|111111111111111|0000|0000000|2222222222222|    4    |000000|000000000|5555555555555|0000000|3333333333333|    4    |000000|000000000|5555555555555|
        # Label: |-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100-100  |0000000000000|-100-100-100-100-100-100-100-100-100-100-100-100|0000000000000|
        ## Common tokens
        prefix_ids = self.tokenizer(self.base_prompt(instruction), add_special_tokens=False)['input_ids']
        common_input_ids = prefix_ids + self.image_id + self.ln_id
        common_modal_ids = [0] * len(prefix_ids) + self.image_modal_ids(1) + [0] * len(self.ln_id)

        ## Seperate tokens
        sep_input_ids_L = self.sep_arm_id + self.image_id + [0] + self.ln_id + self.ln_id + self.sep_action_id
        sep_modal_ids_L = [0] * len(self.sep_arm_id) + self.image_modal_ids(2) + [4] + (2 * len(self.ln_id) + len(self.sep_action_id)) * [0]

        # Action tokens - Left ARM
        readouts = self.config.num_readouts
        if readouts == 0: # We are going to aggregate all the outputs
            label_length = 0
        else:
            sep_input_ids_L += [0] * readouts
            sep_modal_ids_L += [5] * readouts
            label_length = readouts

        ## Seperate tokens
        sep_input_ids_R = self.sep_arm_id + self.image_id + [0] + self.ln_id + self.ln_id + self.sep_action_id
        sep_modal_ids_R = [0] * len(self.sep_arm_id) + self.image_modal_ids(3) + [4] + (2 * len(self.ln_id) + len(self.sep_action_id)) * [0]

        # Action tokens
        readouts = self.config.num_readouts
        if readouts == 0: # We are going to aggregate all the outputs
            label_length = 0
        else:
            sep_input_ids_R += [0] * readouts
            sep_modal_ids_R += [5] * readouts
            label_length = readouts

        # torch Tensor
        if self.config.enable_moe:
            ci_ids = torch.tensor([0] * len(common_input_ids) + [1] * len(sep_input_ids_L) + [2] * len(sep_input_ids_R))
            modal_ids = torch.tensor(common_modal_ids + sep_modal_ids_L + sep_modal_ids_R)
            input_ids = torch.tensor(common_input_ids + sep_input_ids_L + sep_input_ids_R)
            label_ids = torch.tensor([0] * (len(common_input_ids) + len(sep_input_ids_L) - label_length) + [1] * label_length + [0] * (len(sep_input_ids_R) - label_length) + [1] * label_length)
            attention_mask = torch.tensor([1] * len(input_ids))
        else:
            common_modal_ids_dup = [0] * len(prefix_ids) + self.image_modal_ids(11) + [0] * len(self.ln_id)
            ci_ids = torch.tensor([1] * len(common_input_ids) + [1] * len(sep_input_ids_L) + [2] * len(common_input_ids) + [2] * len(sep_input_ids_R))
            modal_ids = torch.tensor(common_modal_ids + sep_modal_ids_L + common_modal_ids_dup + sep_modal_ids_R)
            input_ids = torch.tensor(common_input_ids + sep_input_ids_L + common_input_ids + sep_input_ids_R)
            label_ids = torch.tensor([0] * (len(common_input_ids) + len(sep_input_ids_L) - label_length) + [1] * label_length + [0] * (len(common_input_ids) + len(sep_input_ids_R) - label_length) + [1] * label_length)
            attention_mask = torch.tensor([1] * len(input_ids))

        # image preprocessing
        pixel_values_primary = self.process_image(image)
        pixel_values_wrist_r = self.process_image(image_wrist_r)
        pixel_values_wrist_l = self.process_image(image_wrist_l)

        assert len(ci_ids) == len(modal_ids) == len(input_ids) == len(label_ids), f"{len(ci_ids)},{len(modal_ids)},{len(input_ids)},{len(label_ids)}, {ci_ids},{modal_ids},{input_ids},{label_ids}"
        
        return dict(
            ci_ids=ci_ids,
            modal_ids=modal_ids,
            input_ids=input_ids,
            label_ids=label_ids,
            pixel_values_primary=pixel_values_primary,
            pixel_values_wrist_r=pixel_values_wrist_r,
            pixel_values_wrist_l=pixel_values_wrist_l,
            attention_mask=attention_mask
        )
    
    ###
    def prepare_embeds(self, batch):
        image_embeds_primary = self.image_embeds(batch['pixel_values_primary'])
        image_embeds_wrist_r = self.image_embeds(batch['pixel_values_wrist_r'])
        image_embeds_wrist_l = self.image_embeds(batch['pixel_values_wrist_l'])
        input_embeds = self.language_embeds(batch['input_ids']) 

        # MLP seems to have batch operation error, so I'm just doing manually.
        state_embeds_l = self.encode_state(batch['proprio'][:, :, :self.config.state_dim])
        state_embeds_r = self.encode_state(batch['proprio'][:, :, self.config.state_dim:])
        state_embeds = torch.cat([state_embeds_l, state_embeds_r], axis=0)

        ## Insert modal_embeds
        if self.config.enable_moe:
            input_embeds = self.insert_embeds(input_embeds, image_embeds_primary, batch['modal_ids'], 1)
        else:
            input_embeds = self.insert_embeds(input_embeds, image_embeds_primary, batch['modal_ids'], 1)
            input_embeds = self.insert_embeds(input_embeds, image_embeds_primary, batch['modal_ids'], 11)
        input_embeds = self.insert_embeds(input_embeds, image_embeds_wrist_l, batch['modal_ids'], 2)
        input_embeds = self.insert_embeds(input_embeds, image_embeds_wrist_r, batch['modal_ids'], 3)
        input_embeds = self.insert_embeds(input_embeds, state_embeds, batch['modal_ids'], 4)

        # Insert action_embeds if modeling is regression
        if self.config.modeling == 'regression' or self.config.modeling == 'denoising':
            if self.config.num_readouts != 0 and not self.config.readout_token_as_eos:
                action_embeds_l = self.action_token_l.repeat(input_embeds.shape[0], 1)
                action_embeds_r = self.action_token_r.repeat(input_embeds.shape[0], 1)
                input_embeds = self.insert_embeds(input_embeds, torch.cat([action_embeds_l, action_embeds_r], axis=0), batch['modal_ids'], 5)
        return input_embeds.contiguous(), batch['attention_mask'], batch['ci_ids']
    
    def insert_embeds(self, input_embeds, modal_embeds, modal_ids, target_modal_id):
        modal_mask = modal_ids == target_modal_id
        input_embeds[modal_mask] = modal_embeds.reshape(-1, input_embeds.shape[-1]).to(dtype=input_embeds.dtype)
        return input_embeds
    
    def encode_state(self, state):
        # b, l, dim
        arm_state_embeddings = self.embed_arm_state(state)
        return arm_state_embeddings

    def create_position_ids(self, ci_ids):        
        mask0 = (ci_ids == 0)
        mask1 = (ci_ids == 1)
        mask2 = (ci_ids == 2)
        
        # initial position_ids
        position_ids = torch.zeros_like(ci_ids, dtype=torch.long).to(ci_ids.device)
        
        pos0 = torch.cumsum(mask0.long(), dim=1) - 1
        # modal==0 -> pos0
        position_ids = torch.where(mask0, pos0, position_ids)
        
        # calculate offset
        count0 = torch.sum(mask0.long(), dim=1, keepdim=True)
        
        order_1 = torch.cumsum(mask1.long(), dim=1) - 1  # modal==1 
        order_2 = torch.cumsum(mask2.long(), dim=1) - 1  # modal==2
        
        position_ids = torch.where(mask1, count0 + order_1, position_ids)
        position_ids = torch.where(mask2, count0 + order_2, position_ids)
        
        return position_ids
    
    def create_4d_causal_mask(self, attn_mask_1d: torch.LongTensor, ci_ids=None):
        device = attn_mask_1d.device
        B, seqlen = attn_mask_1d.shape
        # Base causal lower-triangular mask
        base_causal_mask = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool, device=device))
        causal_mask = base_causal_mask.unsqueeze(0).expand(B, -1, -1).clone()  # (B, seqlen, seqlen)

        if ci_ids is not None:
            # Build cross-modal (L->R and R->L) override mask
            ci_key = ci_ids.unsqueeze(1)      # (B, 1, seqlen)
            ci_query = ci_ids.unsqueeze(2)    # (B, seqlen, 1)
            
            LR_mask = (ci_query == 1) & (ci_key == 2)   # Query=L, Key=R
            RL_mask = (ci_query == 2) & (ci_key == 1)   # Query=R, Key=L

            # Length of each modal section (assuming consistent across batch)
            modal_len = (ci_ids[0] == 1).sum().item()
            tri_modal = torch.tril(torch.ones((B, modal_len, modal_len), dtype=torch.bool, device=device))  # (modal_len, modal_len)
            
            # Flatten for assignment using mask
            tri_modal_flat = tri_modal.view(-1)

            # Mask override for L->R and R->L blocks
            causal_mask[LR_mask] = tri_modal_flat
            causal_mask[RL_mask] = tri_modal_flat

            # Apply padding mask (1 means keep)
            valid_mask = attn_mask_1d.bool().unsqueeze(1)      # (B, 1, seqlen)
            valid_mask = valid_mask.expand(-1, seqlen, -1)     # (B, seqlen, seqlen)

            final_mask = causal_mask & valid_mask              # (B, seqlen, seqlen)
            final_mask = final_mask.unsqueeze(1)               # (B, 1, seqlen, seqlen)
        else:
            final_mask = causal_mask.unsqueeze(1)

        twinvla_mask = torch.full((B, 1, seqlen, seqlen), torch.finfo(self.dtype).min, device=device)
        twinvla_mask.masked_fill_(final_mask, 0.0)
        return twinvla_mask

    ## Copied from Qwen2VL
    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    ## Copied from Qwen2VL
    def apply_rotary_pos_emb(self, q, k, cos, sin):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
    
    def fuse_model(self, model_l, model_r, inputs): ## The most naiive way to do task arithmatic
        return (model_l(inputs) + model_r(inputs)) / 2.0
        # return model_l(inputs)

    def in_layernorm_attn(self, idx, inputs, common_mask, left_mask, right_mask):
        ## Layernorm
        #### Common inputs part
        B = inputs.shape[0]
        if self.config.enable_moe:
            common_inputs = inputs[common_mask].view(B, -1, self.hidden_dim())
            common_outputs = self.fuse_model(self.in_layernorm(self.backbone_l, idx), self.in_layernorm(self.backbone_r, idx), common_inputs)
        else:
            common_outputs = None
        #### Independent inputs part
        left_inputs = inputs[left_mask].view(B, -1, self.hidden_dim())
        right_inputs = inputs[right_mask].view(B, -1 , self.hidden_dim())
        left_outputs = self.in_layernorm(self.backbone_l, idx)(left_inputs)
        right_outputs = self.in_layernorm(self.backbone_r, idx)(right_inputs)

        return common_outputs, left_outputs, right_outputs
    
    def out_layernorm_attn(self, idx, inputs, common_mask, left_mask, right_mask):
        ## Layernorm
        #### Common inputs part
        B = inputs.shape[0]
        if self.config.enable_moe:
            common_inputs = inputs[common_mask].view(B, -1, self.hidden_dim())
            common_outputs = self.fuse_model(self.out_layernorm(self.backbone_l, idx), self.out_layernorm(self.backbone_r, idx), common_inputs)
        else:
            common_outputs = None
        #### Independent inputs part
        left_inputs = inputs[left_mask].view(B, -1, self.hidden_dim())
        right_inputs = inputs[right_mask].view(B, -1 , self.hidden_dim())
        left_outputs = self.out_layernorm(self.backbone_l, idx)(left_inputs)
        right_outputs = self.out_layernorm(self.backbone_r, idx)(right_inputs)

        return common_outputs, left_outputs, right_outputs
    
        ## Copied from Qwen2VL
    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    # Attention Re-weighting
    def apply_modality_mask(self, attn_weights, ci_ids, scale_factor):
        """
        attn_weights: [B, H, T, T]
        ci_ids: [B, T]
        """
        B, H, T, _ = attn_weights.shape
        q_ids = ci_ids[:, :, None]  # [B, T, 1]
        k_ids = ci_ids[:, None, :]  # [B, 1, T]

        # Mask for shared modalites
        match_mask = (k_ids != 0).unsqueeze(1).expand(-1, H, T, -1)  # [B, H, T, T]

        with torch.no_grad():
            re_weights = attn_weights * (match_mask + scale_factor * (~match_mask))
            re_weights = re_weights / re_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        return attn_weights + (re_weights - attn_weights).detach()


    def eager_attention_foward(self, module, query, key, value, attention_mask, scaling, ci_ids=None, dropout=0.0):
        key_states = self.repeat_kv(key, module.num_key_value_groups)
        value_states = self.repeat_kv(value, module.num_key_value_groups)

        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)

        if self.config.enable_moe and self.config.attn_reweighting and ci_ids is not None:
            attn_weights = self.apply_modality_mask(attn_weights, ci_ids, 2.0)

        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training).to(value_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states).transpose(1, 2).contiguous()
        return attn_output, attn_weights
    
    def moe_mlp(self, idx, common_inputs, output_weights=False):
        logits = self.gates[idx](common_inputs)
        scores = F.softmax(logits, dim=-1)
        expert_idx = torch.argmax(scores, dim=-1)

        input_shape_prefix = scores.shape[:-1]
        expert_idx_flat = expert_idx.view(-1)

        hard_mask_flat = F.one_hot(expert_idx_flat, num_classes=2).float()
        hard_mask = hard_mask_flat.view(*input_shape_prefix, 2)

        ste_mask = (hard_mask - scores).detach() + scores
        ste_mask_expanded = ste_mask.unsqueeze(-2)

        output_l = self.mlp(self.backbone_l, idx)(common_inputs)
        output_r = self.mlp(self.backbone_r, idx)(common_inputs)
        all_outputs = torch.stack([output_l, output_r], dim=-1)
        masked_outputs = all_outputs * ste_mask_expanded
        
        outputs = masked_outputs.sum(dim=-1)
        return outputs, scores if output_weights else None
    
    # Based on Qwen2VL
    def twinvla_attention(self, input_embeds, attention_mask, ci_ids, output_attn=False):
        # Prepare inputs
        hidden_states = input_embeds

        position_ids = self.create_position_ids(ci_ids)
        position_embeds = self.pos_embed(hidden_states, position_ids)
        attention_mask = self.create_4d_causal_mask(attention_mask, ci_ids)

        # Prepare ci masks
        common_mask = (ci_ids == 0)
        left_mask = (ci_ids == 1)
        right_mask = (ci_ids == 2)

        B = input_embeds.shape[0]
        device = input_embeds.device

        if output_attn:
            moe_weights = []

        for i in range(24): ## Hard coded
            residual = hidden_states
            input_shape = hidden_states.shape[:-1]
            #### In layernorm
            common_outputs, left_outputs, right_outputs = self.in_layernorm_attn(i, hidden_states, common_mask, left_mask, right_mask)
            #### Common inputs part
            if self.config.enable_moe:
                common_hidden_shape = (*common_outputs.shape[:-1], -1, self.head_dim)
                common_Q = self.fuse_model(self.attn(self.backbone_l, i).q_proj, self.attn(self.backbone_r, i).q_proj, common_outputs).view(common_hidden_shape).transpose(1, 2)
                common_K = self.fuse_model(self.attn(self.backbone_l, i).k_proj, self.attn(self.backbone_r, i).k_proj, common_outputs).view(common_hidden_shape).transpose(1, 2)
                common_V = self.fuse_model(self.attn(self.backbone_l, i).v_proj, self.attn(self.backbone_r, i).v_proj, common_outputs).view(common_hidden_shape).transpose(1, 2)

            #### Independent inputs part
            left_hidden_shape = (*left_outputs.shape[:-1], -1, self.head_dim)
            right_hidden_shape = (*right_outputs.shape[:-1], -1, self.head_dim)
            left_Q = self.attn(self.backbone_l, i).q_proj(left_outputs).view(left_hidden_shape).transpose(1, 2)
            left_K = self.attn(self.backbone_l, i).k_proj(left_outputs).view(left_hidden_shape).transpose(1, 2)
            left_V = self.attn(self.backbone_l, i).v_proj(left_outputs).view(left_hidden_shape).transpose(1, 2)
            right_Q = self.attn(self.backbone_r, i).q_proj(right_outputs).view(right_hidden_shape).transpose(1, 2)
            right_K = self.attn(self.backbone_r, i).k_proj(right_outputs).view(right_hidden_shape).transpose(1, 2)
            right_V = self.attn(self.backbone_r, i).v_proj(right_outputs).view(right_hidden_shape).transpose(1, 2)

            ## Attention
            if self.config.enable_joint_attn:
                if self.config.enable_moe:
                    q_embed = torch.cat([common_Q, left_Q, right_Q], axis=2)
                    k_embed = torch.cat([common_K, left_K, right_K], axis=2)
                    v_embed = torch.cat([common_V, left_V, right_V], axis=2)
                else:
                    q_embed = torch.cat([left_Q, right_Q], axis=2)
                    k_embed = torch.cat([left_K, right_K], axis=2)
                    v_embed = torch.cat([left_V, right_V], axis=2)

                cos, sin = position_embeds
                q_embed, k_embed = self.apply_rotary_pos_emb(q_embed, k_embed, cos, sin)

                attn_output, _ = self.eager_attention_foward(
                    self.attn(self.backbone_l, i),
                    query=q_embed,
                    key=k_embed,
                    value=v_embed,
                    attention_mask=attention_mask,
                    ci_ids=ci_ids,
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling
                )
            else:
                # Okay, do not split naiively half....
                q_embed = torch.cat([left_Q, right_Q], axis=2)
                k_embed = torch.cat([left_K, right_K], axis=2)
                v_embed = torch.cat([left_V, right_V], axis=2)

                cos, sin = position_embeds
                q_embed, k_embed = self.apply_rotary_pos_emb(q_embed, k_embed, cos, sin)

                length = q_embed.shape[2] // 2

                left_q_embed = q_embed[:, :, :length]
                left_k_embed = k_embed[:, :, :length]
                left_v_embed = v_embed[:, :, :length]
                right_q_embed = q_embed[:, :, length:]
                right_k_embed = k_embed[:, :, length:]
                right_v_embed = v_embed[:, :, length:]

                # print(attention_mask.shape)
                attention_mask = attention_mask[:, :, :length, :length]

                attn_output_left, _ = self.eager_attention_foward(
                    self.attn(self.backbone_l, i),
                    query=left_q_embed,
                    key=left_k_embed,
                    value=left_v_embed,
                    attention_mask=attention_mask,
                    ci_ids=None, # Please make None for TwinVLA-Scratch experiments!!!
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling
                )
                attn_output_right, _ = self.eager_attention_foward(
                    self.attn(self.backbone_l, i),
                    query=right_q_embed,
                    key=right_k_embed,
                    value=right_v_embed,
                    attention_mask=attention_mask,
                    ci_ids=None, # Please make None for TwinVLA-Scratch experiments!!!
                    dropout=0.0 if not self.training else self.attention_dropout,
                    scaling=self.scaling
                )

                attn_output = torch.cat([attn_output_left, attn_output_right], dim=1)

            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            ## Common inputs part
            if self.config.enable_moe:
                common_attn_output = attn_output[common_mask].view(B, -1, self.hidden_dim())
                common_attn_output = self.fuse_model(self.attn(self.backbone_l, i).o_proj, self.attn(self.backbone_r, i).o_proj, common_attn_output)
            ## Independent inputs part
            left_attn_output = attn_output[left_mask].view(B, -1, self.hidden_dim())
            right_attn_output = attn_output[right_mask].view(B, -1, self.hidden_dim())
            left_attn_output = self.attn(self.backbone_l, i).o_proj(left_attn_output)
            right_attn_output = self.attn(self.backbone_r, i).o_proj(right_attn_output)

            ## Concatenate Again
            if self.config.enable_moe:
                attn_output = torch.cat([common_attn_output, left_attn_output, right_attn_output], axis=1)
            else:
                attn_output = torch.cat([left_attn_output, right_attn_output], axis=1)
            hidden_states = residual + attn_output

            ## Out layernorm
            residual = hidden_states
            common_outputs, left_outputs, right_outputs = self.out_layernorm_attn(i, hidden_states, common_mask, left_mask, right_mask)
            ## MoE
            if self.config.enable_moe:
                common_outputs, moe_weight = self.moe_mlp(i, common_outputs, output_attn)
            left_outputs = self.mlp(self.backbone_l, i)(left_outputs)
            right_outputs = self.mlp(self.backbone_r, i)(right_outputs)
            ## Concatenate Again

            if self.config.enable_moe:
                attn_output = torch.cat([common_outputs, left_outputs, right_outputs], axis=1)
            else:
                attn_output = torch.cat([left_outputs, right_outputs], axis=1)
            hidden_states = residual + attn_output

            if self.config.enable_moe and output_attn:
                # attn_weights.append(attn_weight.detach().cpu().numpy())
                moe_weights.append(moe_weight.detach().cpu().numpy())

        # Final layernorm
        hidden_states = self.norm(hidden_states)
        # return hidden_states, attn_weights if output_attn else None
        return hidden_states, moe_weights if (self.config.enable_moe and output_attn) else None
    
    def base_forward(self, batch, output_attn=False):
        batch = batch.to(device=self.device)
        inputs_embeds, attention_mask, ci_ids = self.prepare_embeds(batch)
        outputs, attn_weights = self.twinvla_attention(inputs_embeds, attention_mask, ci_ids, output_attn=output_attn)
        outputs = self.forward_action_head(outputs, batch)
        return outputs, attn_weights if output_attn else None

    def forward_action_head(self, outputs, batch):
        output = {}

        bsz = batch['modal_ids'].shape[0]
        hidden_state = outputs
        if self.config.num_readouts == 0:
            action_token_label = batch['attention_mask']
        else:
            action_token_label = batch['modal_ids'] == 5

        action_token = hidden_state[action_token_label].reshape(-1, self.config.num_readouts, self.hidden_dim()) # 2B, readout, D
        single_action_shape = (-1, self.config.action_len, self.config.action_dim)
        single_proprio_shape = (-1, self.config.state_dim)
        action_token = self.agg(action_token)

        batched_action = batch['action'].view(*single_action_shape)
        batched_proprio = batch['proprio'][:, 0, :].view(*single_proprio_shape)

        if self.config.share_decoder:
            loss = self.decoder_c(batched_action, action_token, batched_proprio) # 2B, 1, D
        else:
            loss_l = self.decoder_l(batched_action[:bsz], action_token[:bsz], batched_proprio[:bsz])
            loss_r = self.decoder_r(batched_action[bsz:], action_token[bsz:], batched_proprio[bsz:])
            loss = (loss_l + loss_r) / 2.0
        output['loss'] = loss

        return output
    
    def inference_action_head(self, logits=None, hidden_states=None, states=None, action_ids=None, action_len=None, action_dim=None, cfg=1.0):
        if self.config.share_decoder:
            action_head_dtype = self.decoder_c.dtype
            normalized_action = self.decoder_c.denoise(hidden_states.to(dtype=action_head_dtype), states.to(dtype=action_head_dtype), cfg=cfg)
        else:
            bsz = hidden_states.shape[0] // 2
            action_head_dtype = self.decoder_l.dtype
            action_l = self.decoder_l.denoise(hidden_states[:bsz].to(dtype=action_head_dtype), states[:bsz].to(dtype=action_head_dtype), cfg=cfg)
            action_r = self.decoder_r.denoise(hidden_states[bsz:].to(dtype=action_head_dtype), states[bsz:].to(dtype=action_head_dtype), cfg=cfg)
            normalized_action = torch.cat([action_l, action_r], axis=-1)
        return normalized_action
    
    def inference(self, batch, action_len=16, action_dim=10, cfg=1.0, output_attn=False, output_action_token=False):
        bsz = batch['modal_ids'].shape[0]

        inputs_embeds, attention_mask, ci_ids = self.prepare_embeds(batch)
        outputs, attn_weights = self.twinvla_attention(inputs_embeds, attention_mask, ci_ids, output_attn=output_attn)
        hidden_state = outputs
        if self.config.num_readouts == 0:
            action_token_label = batch['attention_mask']
        else:
            action_token_label = batch['modal_ids'] == 5
        action_token = hidden_state[action_token_label].reshape(-1, self.config.num_readouts, self.hidden_dim())
        proprio = batch['proprio'][:, 0, :].reshape(-1, self.config.state_dim)
        if self.config.share_decoder:
            action_head_dtype = self.decoder_c.dtype
        else:
            action_head_dtype = self.decoder_l.dtype
        
        action_token = self.agg(action_token)

        normalized_action = self.inference_action_head(
            hidden_states=action_token.to(dtype=action_head_dtype), 
            states=proprio.to(dtype=action_head_dtype), 
            action_len=self.config.action_len,
            cfg=cfg
        )
        normalized_action = normalized_action.reshape(bsz, action_len, -1)
        if output_attn:
            return normalized_action, attn_weights
        if output_action_token:
            return normalized_action, action_token
        else:
            return normalized_action, None

    @classmethod
    def _from_config(cls, config, **kwargs):
        return cls._construct_from_config(config, **kwargs)

    @classmethod
    def _construct_from_config(cls, config, **kwargs):
        return cls(config, **kwargs)
    
    def module_pop(self, from_model, target_name):
        target_before = target_name.split('.')[:-1]
        if len(target_before) == 0:
            attr_target = from_model
        else:
            attr_target = None
        for tgt in target_before:
            if attr_target is None:
                attr_target = getattr(from_model, tgt)
            else:
                attr_target = getattr(attr_target, tgt)
        target = target_name.split('.')[-1]
        return attr_target._modules.pop(target)
        
    def construct_twinvla(self, singlevla_config_path):
        self.singlevla_config = AutoConfig.from_pretrained(singlevla_config_path)
        singlevla_pretrained_path = self.config.singlevla_pretrained_path
        if singlevla_pretrained_path is None:
            singlevla_pretrained_path = self.singlevla_config.pretrained_path
        revision = getattr(self.singlevla_config, "revision", "main")
        singlevla = AutoModel.from_pretrained(
            singlevla_pretrained_path,
            config=self.singlevla_config,
            low_cpu_mem_usage=False,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.config.singlevla_config = self.singlevla_config
        # build twinvla from singlevla

        # copy norm and pos_embed
        self._modules['norm'] = self.module_pop(singlevla, self.copy_norm_target())
        self._modules['pos_embed'] = self.module_pop(singlevla, self.copy_pos_embed_target())

        if not self.config.share_embed_tokens:
            self._modules['embed_tokens_r'] = self.module_pop(singlevla, self.copy_embed_tokens_target())
            self.embed_tokens_l = copy.deepcopy(self.embed_tokens_r)
        else:
            self._modules['embed_tokens_c'] = self.module_pop(singlevla, self.copy_embed_tokens_target())

        self._modules['backbone_r'] = self.module_pop(singlevla, self.copy_backbone_target())
        self.backbone_l = copy.deepcopy(self.backbone_r)
        
        if not self.config.share_vision:
            self._modules['vision_r'] = self.module_pop(singlevla, self.copy_vision_target())
            self.vision_l = copy.deepcopy(self.vision_model_r)
        else:
            self._modules['vision_c'] = self.module_pop(singlevla, self.copy_vision_target())

        if not self.config.share_decoder:
            self._modules['decoder_r'] = self.module_pop(singlevla, 'action_head')
            self.decoder_l = copy.deepcopy(self.decoder_r)
        else:
            if self.config.dit_scratch:
                print('Init DiT newly for Ablation')
                print(self.singlevla_config)
                self._modules['decoder_c'] = DiTPolicy(
                    model_type='DiT-B',
                    token_size=self.hidden_dim(), 
                    in_channels=self.singlevla_config.action_dim, 
                    state_dim=self.singlevla_config.state_dim,
                    future_action_window_size=self.singlevla_config.action_len,
                    hidden_dim=self.singlevla_config.action_head_hidden_dim,
                    diffusion_steps=self.singlevla_config.train_denoising_steps,
                    test_denoising_steps=self.singlevla_config.test_denoising_steps,
                    denoiser=self.singlevla_config.denoiser,
                    enable_cfg=self.singlevla_config.enable_cfg
                )
                print(self._modules['decoder_c'])
            else:
                self._modules['decoder_c'] = self.module_pop(singlevla, 'action_head')

        if self.singlevla_config.num_readouts != 0 and not self.singlevla_config.readout_token_as_eos:
            self.action_token_r = nn.Parameter(singlevla.action_token.detach().clone()).to(self.device)
            self.action_token_r.require_grad = True
            self.action_token_l = nn.Parameter(singlevla.action_token.detach().clone()).to(self.device)
            self.action_token_l.require_grad = True

        if self.copy_other_target() != None:
            for tgt in self.copy_other_target():
                self._modules[tgt] = self.module_pop(singlevla, tgt)

        self.embed_arm_state = self.module_pop(singlevla, 'embed_arm_state')

        self.agg = self.module_pop(singlevla, 'agg')

        if self.config.enable_moe:
            gate_list = []
            for i in range(24):
                gate_list.append(nn.Linear(self.hidden_dim(), 2, dtype=torch.bfloat16))
            self.gates = nn.ModuleList(gate_list)

        for tgt in ['action_dim', 'action_len', 'state_dim', 'normalization', 'modeling', 'action_head', 'num_readouts', 'readout_token_as_eos', 'denoiser']:
            setattr(self.config, tgt, getattr(singlevla.config, tgt))
        del singlevla

        ## save some parameters for joint attention
        self.head_dim = self.attn(self.backbone_l, 0).head_dim
        self.attention_dropout = self.attn(self.backbone_l, 0).attention_dropout
        self.scaling = self.attn(self.backbone_l, 0).scaling
        self.training = True

        ## Config update
        self.init_processor_tokenizer(self.singlevla_config)
        self.prepare_inputs_ids()

        print('TwinVLA init Done')