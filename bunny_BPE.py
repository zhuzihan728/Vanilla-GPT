import os
import json
import regex as re
from utils.utils import ask_to_download
import timeit

CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')
encoder_path = os.path.join(CACHE_DIR, 'encoder.json')
vocab_path = os.path.join(CACHE_DIR, 'vocab.bpe')
encoder_url = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/encoder.json'
vocab_url = 'https://openaipublic.blob.core.windows.net/gpt-2/models/124M/vocab.bpe'

ask_to_download(encoder_url, encoder_path)
ask_to_download(vocab_url, vocab_path)

def byte_x_unicode():
    nice_bytes = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    ugly_bytes = [code for code in range(2**8) if code not in nice_bytes]
    
    byte_encoder = {code: chr(code) for code in nice_bytes}
    byte_encoder.update({code: chr(i+2**8) for i, code in enumerate(ugly_bytes)})
    
    byte_decoder = {v: k for k, v in byte_encoder.items()}
    return byte_encoder, byte_decoder


class BPETokenizer:
    def __init__(self):
        self.byte_encoder, self.byte_decoder = byte_x_unicode()
        with open(encoder_path, 'r') as f:
            self.encoder = json.load(f)
            self.decoder = {v: k for k, v in self.encoder.items()}
            
        with open(vocab_path, 'r', encoding='utf-8') as f:
            vocab = f.read().split('\n')[1:-1]
            self.merge_ranks = {tuple(pair.split()): i for i, pair in enumerate(vocab)}
            
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.cache = {}
        print(f"=== Bunny Tokenizer ===")
        print(f"===> vocab size: {self.vocab_size}")
    
    @property
    def vocab_size(self):
        return len(self.encoder)
        
    def get_pairs(self, word):
        pairs = []
        prev_char = word[0]
        for char in word[1:]:
            pairs.append((prev_char, char))
            prev_char = char
        return pairs
    
    def merge(self, chunk_unicode):
        if chunk_unicode in self.cache:
            # If the chunk is in the cache, return it
            return self.cache[chunk_unicode]
        
        pairs = self.get_pairs(chunk_unicode)
        if not pairs:
            # If the chunk is a single unit, return it
            return chunk_unicode
        
        top_rank = sorted(pairs, key=lambda x: self.merge_ranks.get(x, float('inf')))[0]
        if top_rank not in self.merge_ranks:
            # no new merge is possible, return the chunk
            self.cache[chunk_unicode] = chunk_unicode
            return chunk_unicode
        
        new_chunk = tuple()
        previous_merge = False
        for pair in pairs:
            if previous_merge:
                previous_merge = False
                continue
            if pair == top_rank:
                new_chunk += ("".join(pair),)
                previous_merge = True
            else:
                new_chunk += (pair[0],)
        
        if not previous_merge:
            new_chunk += (pair[1],)
        
        return self.merge(new_chunk)

    def encode(self, text):
        text_chunks = self.pat.findall(text)
        token_ids = []
        for chunk in text_chunks:
            chunk_byte = chunk.encode('utf-8')
            chunk_unicode = tuple([self.byte_encoder[code] for code in chunk_byte])
            chunk_merged = self.merge(chunk_unicode)
            chunk_ids = [self.encoder[token] for token in chunk_merged]
            token_ids.extend(chunk_ids)
        return token_ids
    
    def decode(self, token_ids):
        text_unicode = [self.decoder[token] for token in token_ids]
        text = "".join(text_unicode)
        text = bytearray([self.byte_decoder[char] for char in text]).decode('utf-8', errors='replace')
        return text

if __name__ == "__main__":
    from transformers import AutoTokenizer

    openai_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    # openai_tokenizer.config.pad_token_id = openai_tokenizer.config.eos_token_id # suppress a warning

    bunny_tokenizer = BPETokenizer()

    text = "Hello!! I'm mechaBunny19c. It's 2024. G00d day :D 🤗"

    openai_encoded = openai_tokenizer(text).input_ids
    bunny_encoded = bunny_tokenizer.encode(text)
    assert openai_encoded == bunny_encoded, "Bunny encoding is different from OpenAI's encoding."
    
    openai_decoded = openai_tokenizer.decode(openai_encoded)
    bunny_decoded = bunny_tokenizer.decode(bunny_encoded)
    assert openai_decoded == bunny_decoded, "Bunny decoding is different from OpenAI's decoding."
            




    



