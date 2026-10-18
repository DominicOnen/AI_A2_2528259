# AI use disclosure

**Student:** Dominic Onen, 25/28259
**Assignment:** SWE 3513 Individual Assignment 2, Musanze Harvest Intelligence Service (AI_A2_2528259)

## What AI was used

An AI assistant, **Claude (Anthropic)**, was used in a chat session to help produce this submission.
It generated the following.

| Part | What the AI generated |
| --- | --- |
| Source code (`data/`, `src/`, `run_all.py`, `predict.py`, `app.py`) | Data generator, validation, preprocessing, modelling, evaluation, command-line tool and API |
| Tests (`tests/`) | 17 pytest tests, including failure paths |
| Technical report (`docs/`) | The structure and a first draft of the text, with numbers read from the generated artifacts |
| README, test log and declaration templates | First drafts |
| Model-selection rule (simplest model within one standard error) | Suggested and implemented it |

## My own part

I have done the following myself (each box is ticked only if it is true):

- [x ] I read the code and the assignment brief, and I can explain the main steps of the pipeline.
- [ x] I ran `run_all.py`, `predict.py`, `pytest -q` and the API on my own computer.
- [x ] I checked the numbers in the report against the files in `artifacts/`.
- [x ] I rewrote the three interpretations in the report (one regression error, one classification error, one cluster) in my own words.
- [ x] I understand why the simplest model within one standard error was chosen, and I can change a range rule or add a response field when asked.
- [ x] I filled in the test log sections B and C and the README fields from my own computer.
- [ x] I recorded the demonstration video and wrote the handwritten verification card myself.

## What the AI did not do

- It did not run anything on my computer. The test run in section B of the test log is mine.
- It also did not record the video or write the card.
- It did not choose my seed: 8259 comes from my registration number 25/28259.

## Checks that were run during development (by the AI, in its own environment)

- `python run_all.py --student-id ug2528259@ines.ac.rw --seed 8259` was run twice. The dataset fingerprint and all result files were identical.
- `pytest -q`: 17 passed.
- The API was started with `uvicorn`, and `/health`, `/model-info` and `/predict` (valid and invalid) were called.

## My own notes



