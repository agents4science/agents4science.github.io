from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:university-of-chicago::CV6o7XC4",
    messages=[
        {"role": "system", "content": "You are the PLANNER agent."},
        {"role": "user", "content": "Design a workflow to measure CO₂ conversion under three temperatures."}
    ]
)
print(response.choices[0].message.content)

