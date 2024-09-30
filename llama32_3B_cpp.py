from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF",
    filename="*q8_0.gguf",
)
output = llm.create_chat_completion(
      messages = [
          {
              "role": "user",
              "content": "What is the capital of France?"
          }
      ]
)

print(output)
