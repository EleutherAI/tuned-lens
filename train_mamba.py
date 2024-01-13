import torch
import sys
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import mamba_ssm
#sys.path.append('tuned-lens/') <- local tuned-lens with the required changes
from tuned_lens.nn.lenses import TunedLens, LogitLens
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from tuned_lens.scripts import train_loop
from tuned_lens.scripts import ingredients
from pathlib import Path

from transformers.configuration_utils import PretrainedConfig
from transformers import PreTrainedModel
from collections import namedtuple
from dataclasses import dataclass, field


#this is so ugly i want to cry
@dataclass
class mambaConfig:

    d_model: int = 1536
    n_layer: int = 48
    vocab_size: int = 50280
    ssm_cfg: dict = None
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8


class MambaConfig(PretrainedConfig):
    model_type = "mamba"
    attribute_map = {"max_position_embeddings": "context_length"}

    def __init__(
        self,
        dmodel=1536,
        vocab_size=50280,
        n_layer=48,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = dmodel
        self.num_hidden_layers = n_layer
        
        super().__init__(**kwargs)

activations = {}
class MambaModel(PreTrainedModel):
    config_class = MambaConfig
    base_model_prefix = "model"
    name_or_path = "mamba"
    name = "mamba-790m"
    def activation_hook(self, module,input, output):
        if len(output)>1:
            output = output[0]
        activations[module] = output
    
    def __init__(self, config: MambaConfig):
        super().__init__(config)
        self.tokenizer=tokenizer
        self.dmodel = config.hidden_size
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers   
        self.model = MambaLMHeadModel(mambaConfig)
        self.model.name_or_path = "mamba"
    def load_state_dict(self, state_dict, strict=False):
        self.model.load_state_dict(state_dict, strict=strict)

    def hook_intermediate(self):
        activation_hook = self.activation_hook
        self.model.backbone.embedding.register_forward_hook(activation_hook)
        for layer in self.model.backbone.layers:
            layer.register_forward_hook(activation_hook)
    def forward(self, input_ids, output_hidden_states=True):
        activations.clear()
        if output_hidden_states==True:
            self.hook_intermediate()
        outputs = self.model(input_ids).logits
        hidden_states=[]
        for layer in activations.keys():
            hidden_states.append(activations[layer])
        hidden_states=hidden_states
        CausalLMOutput = namedtuple("CausalLMOutput", ["logits", "hidden_states"])
        return CausalLMOutput(logits=outputs, hidden_states=hidden_states)

    def get_output_embeddings(self):
        return self.model.lm_head

    def load(self,device):
        return self.to(device), self.tokenizer

device = torch.device('cuda')

model = MambaLMHeadModel.from_pretrained("state-spaces/mamba-790m", device="cuda", dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
config = MambaConfig(dmodel=1536, vocab_size=50280, n_layer=48)
mamba= MambaModel(config)
mamba.load_state_dict(model.state_dict())


train_data = ingredients.Data(["val.jsonl"])
optimizer = ingredients.Optimizer()
distributer = ingredients.Distributed(per_gpu_batch_size=2)
p = Path("my_lens/mamba-790m")
loss=train_loop.LossChoice.KL
train_data.split = "train"
train_data.text_column="text"
train = train_loop.Train(mamba,train_data,optimizer,distributer,p,wandb="Lens",loss=loss)
train.execute()




