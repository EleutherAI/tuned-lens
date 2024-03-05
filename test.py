from transformers import AutoTokenizer,LlamaTokenizer
name = "togethercomputer/StripedHyena-Hessian-7B"
tokenizer = AutoTokenizer.from_pretrained(name,trust_remote_code=True,cache_dir="/mnt/ssd-1/hf_cache/hub",legacy=False)
