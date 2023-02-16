import openai

openai.api_key = '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'

start_sequence = "\nAI:"
restart_sequence = "\nHuman: "

def chat(msg: str) -> str:
  response = openai.Completion.create(
    model="text-davinci-003",
    prompt=f"Human: {msg}\nAI:",
    temperature=0.9,
    max_tokens=1000,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0.6,
    stop=[" Human:", " AI:"]
  )

  return response['choices'][0]['text']
