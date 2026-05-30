"""A transparent ReAct-style agent loop with a typed tool registry."""
from __future__ import annotations
import json, time, traceback
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    schema: dict


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def specs(self) -> List[dict]:
        return [{"type": "function",
                 "function": {"name": t.name, "description": t.description,
                              "parameters": t.schema}}
                for t in self._tools.values()]

    def call(self, name: str, args: dict) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        return self._tools[name].fn(**args)


@dataclass
class Trace:
    steps: List[dict] = field(default_factory=list)
    def log(self, kind: str, payload: Any):
        self.steps.append({"t": round(time.time(), 3), "kind": kind, "payload": payload})


class Agent:
    def __init__(self, llm_client, registry: ToolRegistry, max_steps: int = 8):
        self.llm = llm_client
        self.registry = registry
        self.max_steps = max_steps

    def run(self, goal: str) -> dict:
        trace = Trace()
        messages = [
            {"role": "system",
             "content": "You are a careful agent. Use tools when needed. "
                        "Stop and answer once you have enough information."},
            {"role": "user", "content": goal},
        ]
        for step in range(self.max_steps):
            resp = self.llm.chat.completions.create(
                model="gpt-4o-mini", messages=messages,
                tools=self.registry.specs(), tool_choice="auto", temperature=0.0)
            msg = resp.choices[0].message
            if not msg.tool_calls:
                trace.log("final", msg.content)
                return {"answer": msg.content, "trace": trace.steps}
            messages.append(msg.model_dump())
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                trace.log("tool_call", {"name": tc.function.name, "args": args})
                result = self._safe_call(tc.function.name, args, trace)
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": json.dumps(result)[:4000]})
        trace.log("halt", "max_steps reached")
        return {"answer": None, "trace": trace.steps}

    def _safe_call(self, name, args, trace, retries: int = 2):
        for attempt in range(retries + 1):
            try:
                out = self.registry.call(name, args)
                trace.log("tool_result", {"name": name, "ok": True})
                return out
            except Exception as exc:  # guardrail: never let a tool crash the loop
                trace.log("tool_error", {"name": name, "attempt": attempt,
                                         "error": str(exc)})
                if attempt == retries:
                    return {"error": str(exc), "trace": traceback.format_exc()[:500]}
                time.sleep(0.5 * (attempt + 1))
