from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe
import torch

from mvp_codes.llms.llm_config_helper import LLMParams
from prettytable import PrettyTable

def model_memory_size(model, input_dtype=torch.float32):
    total_params = 0
    total_grads = 0
    for param in model.parameters():
        param_size = param.numel()
        total_params += param_size
        if param.requires_grad:
            total_grads += param_size

    total_buffers = sum(buf.numel() for buf in model.buffers())
    # Size in bytes = (Number of elements) * (Size of each element in bytes)
    # We assume parameters and gradients are stored in the same type as input dtype
    element_size = torch.tensor(0, dtype=input_dtype).element_size()
    total_memory_bytes = (total_params + total_grads + total_buffers) * element_size
    total_memory_gb = total_memory_bytes / (1024**3)
    return total_memory_gb

def model_stats(model):
    '''
    Estimate the model stats
    '''
    order = 1e6
    total_buff = sum(buf.numel() for buf in model.buffers())
    total_param = sum(p.numel() for p in model.parameters())
    total_unique_params = total_param - model.tok_emb.weight.numel()
    total_trainable_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fp32_memory = model_memory_size(model, input_dtype=torch.float32)
    bf16_memory = model_memory_size(model, input_dtype=torch.bfloat16)
    stats = {
        "total_buff": round(total_buff/order,2),
        "total_param": round(total_param/order, 2),
        "total_unique_params": round(total_unique_params/order, 2),
        "total_trainable_param": round(total_trainable_param/order, 2),
        "fp32_memory": round(fp32_memory,2),
        "bf16_memory": round(bf16_memory,2),
    }
    table = PrettyTable()
    table.field_names = ["Metric in Mil/GB", "Value"]
    for key, value in stats.items():
        table.add_row([key, value])
    
    print(table)
    # print("Prams in million and memory in GB\n",stats)
    return stats

class Llama32Tokenizer:
    def __init__(self, tokenizer_model_path:str):
        # self.params = params
        assert Path(tokenizer_model_path).exists()
        meargeable_ranks = load_tiktoken_bpe(tokenizer_model_path)
        pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"

        self.special_tokens = {
            "<|beginoftext|>": 128000,
            "<|endoftext|>": 128001,
            "<|start_header_id|>": 128006,
            "<|end_header_id|>": 128007,
            "<|eot_id|>": 128009,
        }
        self.special_tokens.update({
            f"<|reserved_{i}|>": 128002 + i for i in range(256) if (128002 + i) not in self.special_tokens.values()
        })
        self.tik_model = tiktoken.Encoding(
            name = Path(tokenizer_model_path).name,
            pat_str=pat_str,
            mergeable_ranks=meargeable_ranks,
            special_tokens=self.special_tokens,
        )
    
    def encode(self, text:str, bos: bool=False, eos:bool=False, allowed_special=set(), disallowed_special=()):
        tokens = self.tik_model.encode(text, allowed_special=allowed_special, disallowed_special=disallowed_special)
        if bos:
            tokens += [self.special_tokens['<|begin_of_text|>']]
        if eos:
            tokens += [self.special_tokens['<|end_of_text|>']]
        return tokens

    def decode(self, tokens):
        return self.tik_model.decode(tokens)


class Llama32ChatFormat:
    '''
    chat format:
    message = {
        <|start_header_id|>role(user/system)<|end_header_id|>
        \n\n
        text<|eot_id|>
    }
    '''
    def __init__(self, tokenizer: Llama32Tokenizer, params: LLMParams):
        self.tokenizer = tokenizer
        self.params = params

    def encode_user(self, text:str, bos: bool=False, eos:bool=False):
        tokens = [self.tokenizer.special_tokens['<|start_header_id|>']]
        tokens += self.tokenizer.encode("user", bos=False, eos=False)
        tokens += [self.tokenizer.special_tokens['<|end_header_id|>']]
        tokens += self.tokenizer.encode('\n\n', bos=False, eos=False)
        tokens += self.tokenizer.encode(text.strip(), bos=False, eos=False)
        tokens += [self.tokenizer.special_tokens['<|eot_id|>']]
        return tokens

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def encode_chat(self, chat):
        tokens = []
        for i, (speaker, text) in enumerate(chat):
            if i > 0:
                tokens.append(self.tokenizer.special_tokens['<|eot_id|>'])
            tokens += self.encode(speaker, bos=True)
            tokens += self.encode(text, eos=True)
        return tokens

    def decode_chat(self, tokens):
        chat = []
        speaker = None
        text = ""
        for token in tokens:
            if token == self.tokenizer.special_tokens['<|eot_id|>']:
                chat.append((speaker, text))
                speaker = None
                text = ""
            elif token == self.tokenizer.special_tokens['<|begin_of_text|>']:
                pass
            elif token == self.tokenizer.special_tokens['<|end_of_text|>']:
                pass
            elif speaker is None:
                speaker = self.decode([token])
            else:
                text += self.decode([token])
        if speaker is not None:
            chat.append((speaker, text))
        return chat

    def encode_chats(self, chats):
        return [self.encode_chat(chat) for chat in chats]

    def decode_chats(self, chats):
        return [self.decode_chat(chat) for chat in chats]

