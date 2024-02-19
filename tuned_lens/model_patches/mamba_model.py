from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    AutoTokenizer
)

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf
from mamba_ssm.models.config_mamba import MambaConfig
from collections import namedtuple


class PreMambaConfig(PretrainedConfig):
    model_type = "mamba"
    attribute_map = {"max_position_embeddings": "context_length"}
    def __init__(
        self,
        **kwargs,
    ): 
        super().__init__(**kwargs)


class MambaModel(PreTrainedModel):
    
    config_class = PreMambaConfig
    base_model_prefix = "model"
    def activation_hook(self, module,input, output):
        self.activations[module] = output[0]+output[1]
    
    def __init__(self, config: PreMambaConfig):
        super().__init__(config)
        self.activations = {}
        config_data = load_config_hf(config._name_or_path)
        m_config = MambaConfig(**config_data)
        config.vocab_size = m_config.vocab_size
        config.hidden_size = m_config.d_model
        config.num_hidden_layers = m_config.n_layer
        model = MambaLMHeadModel(m_config)
        self.model = model
 
    def load_state_dict(self, state_dict, strict=False):
        self.model.load_state_dict(state_dict, strict=strict)

    def hook_intermediate(self):
        activation_hook = self.activation_hook
        for layer in self.model.backbone.layers:
            layer.register_forward_hook(activation_hook)

    def forward(self, input_ids, output_hidden_states=True):
        self.activations.clear()
        if output_hidden_states==True:
            self.hook_intermediate()
        outputs = self.model(input_ids).logits
        hidden_states=[]
        for layer in self.activations.keys():
            hidden_states.append(self.activations[layer])
        hidden_states=hidden_states
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits", "hidden_states"])
        return CausalLMOutput(logits=outputs, hidden_states=hidden_states)

    def get_input_embeddings(self):
        return self.model.backbone.embedding
    def get_output_embeddings(self):
        return self.model.lm_head

    def load(self,device):
        return self.to(device), self.tokenizer

class MambaTokenizer(AutoTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def from_pretrained(self, *args, **kwargs):
        return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")