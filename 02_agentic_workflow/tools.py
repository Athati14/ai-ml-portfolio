"""Example tools wired into the agent: calculator + web-ish lookup stub."""
from agent import Tool, ToolRegistry


def make_registry() -> ToolRegistry:
    reg = ToolRegistry()

    reg.register(Tool(
        name="calculator",
        description="Evaluate a basic arithmetic expression.",
        fn=lambda expression: {"result": eval(expression, {"__builtins__": {}}, {})},
        schema={"type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]},
    ))

    def lookup(topic: str) -> dict:
        # Replace with a real search/DB call.
        kb = {"speed of light": "299,792,458 m/s",
              "pi": "3.14159"}
        return {"topic": topic, "value": kb.get(topic.lower(), "unknown")}

    reg.register(Tool(
        name="knowledge_lookup",
        description="Look up a fact by topic name.",
        fn=lookup,
        schema={"type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"]},
    ))
    return reg


if __name__ == "__main__":
    from openai import OpenAI
    from agent import Agent
    agent = Agent(OpenAI(), make_registry())
    print(agent.run("What is the speed of light divided by pi? Use the tools."))
