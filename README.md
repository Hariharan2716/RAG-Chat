# RAG-Chat


## Local Tool-Calling Agent

A small agent that uses a local Ollama model to choose among three safe tools.
It prints each tool call and enforces a maximum number of tool steps.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running
- A model that supports tool calling; this project defaults to `qwen3:4b`

## Setup

1. Install Ollama from https://ollama.com/
2. Open a terminal and download the model:

   ```bash
   ollama pull qwen3:4b
   ```

3. Save/extract this project, enter its folder, and run:

   ```bash
   python3 agent.py
   ```

No Python packages or API keys are required. Ollama runs the model locally.

## Example tasks

- `Calculate (15 + 25 + 40) / 3.`
- `Analyze this text: Agents choose tools. They inspect results!`
- `What is the current date and time?`
- `Find the average of 10, 20, 30, then explain the result.`

## Change the step limit or model

At the top of `agent.py`, change:

```python
MODEL = "qwen3:4b"
MAX_STEPS = 5
```

A step means one tool invocation. The agent cannot execute more than `MAX_STEPS`
tool calls in a run. A final model response does not count as a tool step.

## Safety boundaries

The only available tools are:
1. `calculator`: AST-based arithmetic evaluator (not Python `eval`)
2. `text_analyzer`: counts text statistics
3. `current_datetime`: reads the local system clock

There is no shell, arbitrary code execution, file access, browser, email, or network
tool exposed to the model. The only network connection is the local HTTP connection
to Ollama at `localhost:11434`.

This is a learning prototype, not a production security boundary. Keep Ollama and
the machine itself secured, and review code before extending the tool allowlist.
