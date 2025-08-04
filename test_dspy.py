# %%
# Test for dspy from  https://dspy.ai/learn/programming/language_models/#__tabbed_1_1


# %%
import dspy
import os

endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://openai-eus.openai.azure.com")
key = os.environ.get("AZURE_OPENAI_API_KEY", None)
assert key is not None, "AZURE_OPENAI_API_KEY is not set"
lm = dspy.LM("azure/gpt-4o-mini", api_key=key, api_base=endpoint)
dspy.configure(lm=lm)


lm("Say this is a test!", temperature=0.7)  # => ['This is a test!']
lm(messages=[{"role": "user", "content": "Say this is a test!"}])  # => ['This is a test!']

# %%

# Define a module (ChainOfThought) and assign it a signature (return an answer, given a question).
qa = dspy.ChainOfThought("question -> answer")

# Run with the default LM configured with `dspy.configure` above.
response = qa(question="How many floors are in the castle David Gregory inherited?")
print(response.answer)
# %%
dspy.configure(lm=dspy.LM("azure/gpt-4o-mini"))
response = qa(question="How many floors are in the castle David Gregory inherited?")
print("GPT-4o-mini:", response.answer)
# %%
qa = dspy.ChainOfThought("question -> answer")
model_name = "azure/gpt-4o-mini"
model_name = "azure/o1-mini"
system_prompt = "You are a helpful AI assistant that rewrites questions into well-formed questions."
# model_name = "gpt-4o-mini"  # For local testing, use the model name without "azure/"
text = "which has its runed suitor with his slipshod heels and threadbare dress borrowing and begging through *the* round of every man's acquaintance which gives to mounded might *the* means abundantly of wearying out *the* right"
config = {"temperature": 1, "n": 1, "max_tokens": 20_000, "api_key": key, "api_base": endpoint}
with dspy.context(lm=dspy.LM(model_name, **config)):
    response = qa(question=text, system_prompt=system_prompt)
    print(f"{model_name}: {response.answer}")

# %%
# %%
