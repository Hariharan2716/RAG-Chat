"""
Free Local Tool-Calling Agent
Requires: Python 3.10+ and Ollama running locally.
Only exposes three read-only/safe tools. No shell, filesystem, or network tools.
"""
import ast
import json
import operator
import sys
import urllib.request
from datetime import datetime

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:4b"
MAX_STEPS = 5  # Change this to set the hard tool-call limit.

# ---------- Safe tool implementations ----------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_math(node):
    if isinstance(node, ast.Expression):
        return _safe_math(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left, right = _safe_math(node.left), _safe_math(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 10:
            raise ValueError("Exponent magnitude is limited to 10.")
        result = _ALLOWED_BINOPS[type(node.op)](left, right)
        if abs(result) > 10**100:
            raise ValueError("Result is too large.")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_math(node.operand))
    raise ValueError("Only basic arithmetic is allowed.")


def calculator(expression: str):
    if len(expression) > 200:
        raise ValueError("Expression is too long.")
    tree = ast.parse(expression, mode="eval")
    return {"expression": expression, "result": _safe_math(tree)}


def text_analyzer(text: str):
    words = text.split()
    sentences = sum(text.count(mark) for mark in ".!?")
    return {
        "characters": len(text),
        "words": len(words),
        "sentences_approx": sentences,
    }


def current_datetime():
    return {"local_datetime": datetime.now().astimezone().isoformat(timespec="seconds")}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Safely calculate a basic arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "text_analyzer",
            "description": "Count characters, words, and approximate sentences in text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_datetime",
            "description": "Get the computer's current local date and time.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

FUNCTIONS = {
    "calculator": calculator,
    "text_analyzer": text_analyzer,
    "current_datetime": current_datetime,
}


def ollama_chat(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
    }
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.loads(response.read().decode("utf-8"))["message"]
    except Exception as exc:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_URL}. "
            f"Check that Ollama is running and model '{MODEL}' is installed. Details: {exc}"
        )


def run_agent(task, max_steps=MAX_STEPS):
    system_message = (
        "You are a careful tool-using assistant. Choose tools only when useful. "
        "You have exactly three tools: calculator, text_analyzer, current_datetime. "
        "Never claim to have used a tool unless it appears in the conversation. "
        "You cannot access files, run commands, browse the internet, or perform external actions. "
        "If a task cannot be completed with these tools, explain the limitation. "
        "When you have enough information, give a concise final answer."
    )
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": task},
    ]

    steps_used = 0
    print("\n=== AGENT EXECUTION ===")
    print(f"Task: {task}")
    print(f"Maximum tool steps: {max_steps}\n")

    while True:
        message = ollama_chat(messages)
        messages.append(message)
        tool_calls = message.get("tool_calls") or []

        if not tool_calls:
            print("\n=== FINAL ANSWER ===")
            print(message.get("content", "(No text response.)"))
            print(f"\nTool steps used: {steps_used}/{max_steps}")
            return

        for call in tool_calls:
            if steps_used >= max_steps:
                print("\n[STOP] Maximum step limit reached. No further tools will run.")
                messages.append({
                    "role": "user",
                    "content": "The tool-step limit has been reached. Do not call any more tools. "
                               "Give the best answer possible using only the results already available, "
                               "and clearly state any limitation."
                })
                final_message = ollama_chat(messages)
                print("\n=== FINAL ANSWER (STEP LIMIT) ===")
                print(final_message.get("content", "(No final response.)"))
                print(f"\nTool steps used: {steps_used}/{max_steps}")
                return

            function = call.get("function", {})
            name = function.get("name")
            args = function.get("arguments", {})

            print(f"STEP {steps_used + 1}/{max_steps}")
            print(f"Chosen tool: {name}")
            print(f"Arguments: {json.dumps(args, ensure_ascii=False)}")

            if name not in FUNCTIONS:
                result = {"error": "Tool is not in the permitted allowlist."}
                print(f"Result: {result}\n")
            else:
                try:
                    if not isinstance(args, dict):
                        raise ValueError("Tool arguments must be an object.")
                    result = FUNCTIONS[name](**args)
                except Exception as exc:
                    result = {"error": str(exc)}
                print(f"Result: {json.dumps(result, ensure_ascii=False)}\n")

            steps_used += 1
            messages.append({
                "role": "tool",
                "tool_name": name or "unknown",
                "content": json.dumps(result, ensure_ascii=False),
            })


def main():
    print("Free Local AI Agent (type 'exit' to quit)")
    print(f"Model: {MODEL} | Max tool steps: {MAX_STEPS}")
    while True:
        try:
            task = input("\nYour task: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if task.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if not task:
            continue
        run_agent(task)


if __name__ == "__main__":
    main()
