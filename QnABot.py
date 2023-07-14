from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig
import sys

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
config = GenerationConfig(max_new_tokens=200)

question = "What color is the undoubtedly beautiful sky?"

for question in sys.stdin:
    tokens = tokenizer(question, return_tensors="pt")
    #print(tokens)

    answer = model.generate(**tokens, generation_config=config)
    print(tokenizer.batch_decode(answer, skip_special_tokens=True))
