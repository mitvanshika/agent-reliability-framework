"""
Generates new test scenarios automatically using an LLM, instead of
you hand-writing every question. This directly addresses two things:

1. More than 10 questions — run this to grow the suite as large as
   you want, without manually authoring each one.
2. Removes you as the sole author of every test case — you wrote
   the agent's tools AND all 10 original questions, which risks
   unconsciously writing questions your agent already handles well.
   An LLM generating questions from a description of the tools (not
   from knowing how your agent responds) is a step toward
   independent test authorship.

New scenarios are saved as gen_XX.yaml files alongside your existing
edge_XX.yaml / normal_XX.yaml files — nothing existing is touched or
overwritten. run_suite.py already picks up every .yaml file in
scenarios/, so no other code needs to change.

Usage:
    python generate_scenarios.py 10
    (generates 10 new scenarios; defaults to 10 if no count given)
"""

import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

load_dotenv()

SCENARIOS_DIR = "scenarios"
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "openai/gpt-oss-120b")

AGENT_DESCRIPTION = """The agent is a customer support bot for an online store with these tools:
- order_lookup(order_id): looks up an order by ID (valid IDs: A1001-A1004), returns status/total/days_since_purchase, or "not found" for anything else
- refund_calculator(order_total, days_since_purchase): eligible for full refund only if <= 30 days since purchase
- faq_search(query): only knows about "shipping", "return policy", and "payment methods" — anything else returns "no matching FAQ found"
- escalate_to_human(reason): hands off to a human agent

The agent must decide which tool(s) to call, in what order, and must NOT invent information tools didn't actually provide."""


class GeneratedScenario(BaseModel):
    id: str = Field(description="Short snake_case identifier, e.g. 'gen_multi_intent_01'")
    question: str = Field(description="The customer's message to the agent")
    category: str = Field(description="'normal' for straightforward single-purpose questions, 'edge' for anything ambiguous, multi-part, adversarial, or likely to expose a weakness")


class ScenarioBatch(BaseModel):
    scenarios: list[GeneratedScenario]


llm = ChatGroq(model=GENERATOR_MODEL)
structured_llm = llm.with_structured_output(ScenarioBatch, method="json_mode")


def get_existing_questions() -> list[str]:
    questions = []
    if not os.path.isdir(SCENARIOS_DIR):
        return questions
    for filename in os.listdir(SCENARIOS_DIR):
        if not filename.endswith(".yaml"):
            continue
        with open(os.path.join(SCENARIOS_DIR, filename)) as f:
            for line in f:
                if line.strip().startswith("question:"):
                    q = line.split(":", 1)[1].strip().strip('"')
                    questions.append(q)
    return questions


def get_existing_ids() -> set[str]:
    ids = set()
    if not os.path.isdir(SCENARIOS_DIR):
        return ids
    for filename in os.listdir(SCENARIOS_DIR):
        if filename.endswith(".yaml"):
            ids.add(filename[:-5])
    return ids


def generate_scenarios(count: int) -> list[GeneratedScenario]:
    existing_questions = get_existing_questions()
    existing_list = "\n".join(f"- {q}" for q in existing_questions) if existing_questions else "(none yet)"

    prompt = f"""You are designing test questions to stress-test an AI customer support agent, to catch weaknesses like hallucination, poor tool use, and incomplete answers.

{AGENT_DESCRIPTION}

These questions already exist in the test suite — do NOT generate anything that duplicates or closely resembles these in topic or intent:
{existing_list}

Generate exactly {count} NEW, genuinely different test questions a real customer might ask. Include a mix of:
- Straightforward questions the agent should handle easily (category: normal)
- Questions about topics the agent has NO tool for at all (e.g. payment decline reasons, account login issues, product availability, warranty claims) — these specifically test whether the agent honestly admits it doesn't know or invents an answer (category: edge)
- Multi-part or ambiguous requests that require multiple tools or careful reasoning (category: edge)
- At least one emotionally charged or adversarial question (angry customer, pressure tactics) (category: edge)

Give each a unique snake_case id starting with "gen_".

Respond with a JSON object containing a "scenarios" field: a list of objects, each with "id", "question", and "category" fields."""

    result = structured_llm.invoke(prompt)
    return result.scenarios


def save_scenarios(scenarios: list[GeneratedScenario]):
    os.makedirs(SCENARIOS_DIR, exist_ok=True)
    existing_ids = get_existing_ids()
    saved = 0

    for scenario in scenarios:
        scenario_id = scenario.id
        if scenario_id in existing_ids:
            scenario_id = f"{scenario_id}_v2"

        content = (
            f'id: "{scenario_id}"\n'
            f'question: "{scenario.question.replace(chr(34), chr(39))}"\n'
            f'category: "{scenario.category}"\n'
        )
        path = os.path.join(SCENARIOS_DIR, f"{scenario_id}.yaml")
        with open(path, "w") as f:
            f.write(content)
        print(f"Saved {path}")
        existing_ids.add(scenario_id)
        saved += 1

    return saved


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    print(f"Generating {count} new scenarios using {GENERATOR_MODEL}...")
    scenarios = generate_scenarios(count)
    saved = save_scenarios(scenarios)
    print(f"\nDone — {saved} new scenario files added to {SCENARIOS_DIR}/")
    print("Run 'python pipeline.py' to test both agents against the full (larger) suite.")