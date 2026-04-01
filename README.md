# CTF Agent

## Setup

1. Install: 
```
pip install -e .
ctf-metactf compete.metactf.com/576 --cookie "METACTF_COMPETE=your_token_here"
# or full URL / env cookie
set METACTF_COOKIE=METACTF_COMPETE=...
ctf-metactf https://compete.metactf.com/576
# limit (0 or omit = all unsolved, after sort)
ctf-metactf compete.metactf.com/576 --cookie "..." --limit 4
# skip titles (semicolons)
ctf-metactf compete.metactf.com/576 --cookie "..." --skip "Dead Drop;Great Paywall"
# long cookie with MCS_OPTIONS — only METACTF_* (etc.) is sent; MCS_OPTIONS stripped
```

When multiple keys are present, solver requests are distributed in round-robin order across keys.

## Run

Put your challenge in a directory:

| What | Where |
|------|--------|
| Challenge statement | `challenge.txt`, `description.txt`, `README.md`, or `challenge.md` (first one found wins) |
| Hints | `hints.txt`, `hint.txt`, `hints.md`, or a `hints/` folder with text files |
| Attachments | Any files anywhere in the folder (or use a `distfiles/` subfolder — both work) |
| Remote service | Optional `connection.txt` with `nc ...` or a URL |

Then:

```bash
ctf-solve path/to/that/folder
```

If the folder is named `challenge` in the current directory, you can run:

```bash
ctf-solve
```

Default model lineup:

- `qwen/qwen3.6-plus-preview:free`
- `nvidia/nemotron-3-super-120b-a12b:free`
- `stepfun/step-3.5-flash:free`

Run just one model with:

```bash
ctf-solve path/to/challenge --model openrouter/qwen/qwen3.6-plus-preview:free
```

You can also omit the prefix: `--model qwen/qwen3.6-plus-preview:free`.

Include Gemini via direct Gemini API:

```bash
ctf-solve ./my-challenge --gemini
```

Single-model Gemini:

```bash
ctf-solve ./my-challenge --model gemini/gemini-flash-latest
```

Check all configured keys quickly:

```bash
ctf-solve --check-keys --model openrouter/qwen/qwen3.6-plus-preview:free
```

When a flag is found, it prints in the terminal. You submit it to the competition site yourself if there is one.

### Optional: many challenges at once

```bash
ctf-solve --watch path/to/parent
```

Each **subfolder** of `parent` is one challenge. A coordinator process manages them (Ctrl+C to stop). This is only needed if you are juggling multiple challenges; most of the time use a single folder and `ctf-solve` as above.

### Dry run

```bash
ctf-solve ./my-challenge --no-submit
```

### Debug one model's responses

```bash
export CTF_AGENT_DEBUG_MODEL="qwen/qwen3.6-plus-preview"
ctf-solve ./my-challenge
# disable
unset CTF_AGENT_DEBUG_MODEL
```


## Requirements

- Python 3.13+
- Docker
- OpenRouter API key

## Acknowledgements

- [Pydantic AI OpenRouter](https://ai.pydantic.dev/models/openrouter/)
