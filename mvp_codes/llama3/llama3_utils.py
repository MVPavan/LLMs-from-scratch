from pathlib import Path
import tiktoken
from tiktoken.load import load_tiktoken_bpe

from mvp_codes.llama3.llama3_config import Llama32Params

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
    def __init__(self, tokenizer: Llama32Tokenizer, params: Llama32Params):
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


