"""Tag each validation pair with how the implausible hypothesis differs (Claude API).

Used only for error analysis; labels never feed into training or model selection.
"""
import asyncio
import csv
from enum import Enum
from pathlib import Path

import anthropic
from dotenv import load_dotenv
from pydantic import BaseModel

from art_nlp.data import load_split

MODEL = "claude-opus-5-5"
OUT = Path("results/edit_types_validation.csv")


class EditType(str, Enum):
    negation = "negation"
    entity = "entity"  # different person, object or place
    action = "action"  # different event or verb
    outcome = "outcome"  # opposite result or sentiment
    time_quantity = "time_quantity"
    other = "other"


class Label(BaseModel):
    edit_type: EditType
    absurd_alone: bool  # implausible even without the observations


PROMPT = """Story start: {o1}
Story end: {o2}
Plausible middle: {pos}
Implausible middle: {neg}

Classify how the implausible middle differs from the plausible one:
- negation: adds or removes a negation
- entity: a different person, object or place
- action: a different event or verb
- outcome: an opposite result, emotion or sentiment
- time_quantity: different time, order, number or amount
- other: anything else, or a complete rewrite
absurd_alone: true if the implausible middle is odd or nonsensical on its own,
without reading the story start and end."""


async def label(client, sem, i, ex):
    pos, neg = ex.hyps[ex.label], ex.hyps[1 - ex.label]
    async with sem:
        r = await client.messages.parse(
            model=MODEL, max_tokens=2000, output_config={"effort": "low"}, output_format=Label,
            messages=[{"role": "user", "content": PROMPT.format(o1=ex.obs1, o2=ex.obs2, pos=pos, neg=neg)}])
    if r.stop_reason == "refusal" or r.parsed_output is None:
        return i, None, None
    return i, r.parsed_output.edit_type.value, r.parsed_output.absurd_alone


async def main():
    load_dotenv()
    data = load_split("validation")
    client, sem = anthropic.AsyncAnthropic(), asyncio.Semaphore(16)
    rows = sorted(await asyncio.gather(*(label(client, sem, i, ex) for i, ex in enumerate(data))))
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "edit_type", "absurd_alone"])
        w.writerows(rows)
    print(f"{OUT}: {sum(r[1] is not None for r in rows)}/{len(rows)} labelled")


if __name__ == "__main__":
    asyncio.run(main())
