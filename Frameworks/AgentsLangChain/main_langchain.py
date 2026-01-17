from pipeline.roles import build_roles

def main():
    goal = "Find catalysts that improve CO₂ conversion at room temperature."
    agents = build_roles()

    state = {"goal": goal}
    for agent in agents:
        print(f"\n=== {agent.name} ===")
        output = agent.act(state["goal"])
        print(f"{agent.name} output:\n{output[:400]}\n")
        state["goal"] = output

if __name__ == "__main__":
    main()
