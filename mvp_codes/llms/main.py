from .llm_config_helper import llm_config, get_llm_params
from .llm_utils import Llama32Tokenizer, Llama32ChatFormat, model_stats
from .llm_data_loader import LlamaDLPreTokenized
from .llm_models import Llama32Model


lm32_params = get_llm_params(llm_config.llama32)
lm32 = Llama32Model(params=lm32_params)
stats = model_stats(lm32)
print("here")