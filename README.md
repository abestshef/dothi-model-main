# Dothistroma nursery model

A [Shiny for Python](https://shiny.posit.co/py/) app that reproduces the
interactive Jupyter notebook `Dothi_model_for_RSE.ipynb`: a stochastic **SEI**
(Susceptible → Exposed/asymptomatic → Infected/symptomatic) simulation of
*Dothistroma septosporum* spread through a forest nursery, with four control
strategies compared side by side.

<!-- Choose parameters in the sidebar, click "Run the model", and the two
time-course plots plus a 4×3 grid of field snapshots appear. -->

## Install & run

From this folder (`dothistroma-nursery/`).

### With [uv](https://docs.astral.sh/uv/) (recommended)

```bash
uv sync                       # create .venv and install the app
uv run dothi-nursery          # launch, opens http://127.0.0.1:8000 in your browser
```

Options: `uv run dothi-nursery --port 8080 --no-browser`.

Live-reload while editing, straight through Shiny:

```bash
uv run shiny run --reload src/dothi_nursery/app.py:app
```

### With pip / venv

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
dothi-nursery                      # or: shiny run --reload src/dothi_nursery/app.py:app
```

## What the model does

- Saplings sit on a 0.5 m × 0.5 m grid (30 wide × 25 deep), with every 5th row
  left empty. Row 0 is an **external source** of infection along the southern edge.
- Infection spreads plant-to-plant via a distance kernel (95% of infections
  within ~2 m) and is **seasonal** — no spread in winter. New infections are
  asymptomatic for an incubation period before becoming symptomatic and
  infectious. The simulation is a Gillespie stochastic process.
- At the chosen **control month** the field is surveyed and four strategies run
  from the same starting state:
  1. Remove symptomatic only, **replace** with healthy saplings
  2. Remove symptomatic only, **leave empty**
  3. Remove symptomatic **+ a radius**, replace
  4. Remove symptomatic **+ a radius**, leave empty

### Parameters

| Control | Meaning |
| --- | --- |
| Within-field infection rate | Low / Moderate / High plant-to-plant transmission |
| External infection rate | Strength of the woodland source along the edge |
| Average incubation period | 2 / 4 / 6 months asymptomatic before symptoms |
| Control month | When the field is surveyed and treated (6–24) |
| Control radius | Removal radius in metres for strategies 3 & 4 |

## Outputs

- **Time courses** — symptomatic cases, and symptomatic + removed ("yield loss"),
  over 30 months, with the four strategies overlaid, seasonal shading, and the
  control month marked.
- **Field snapshots** — a 4×3 grid: one row per strategy, columns showing the
  field just before control, just after, and at 30 months. Blue = susceptible,
  plum = asymptomatic, red = symptomatic, black = empty/removed.

## Project layout

```
src/dothi_nursery/
  model.py      # the stochastic simulation (pure, importable, vectorised)
  plotting.py   # the two matplotlib figures
  app.py        # the Shiny UI + server
  cli.py        # the `dothi-nursery` launcher
tests/
  test_model.py # smoke tests (pytest)
```

The model logic is faithful to the notebook; the dispersal kernel is vectorised
with NumPy so a run takes a fraction of a second rather than up to a minute.
Because the model is stochastic, the outcome varies on each run (pass a seeded
`numpy.random.Generator` to `run_simulation` for reproducibility).

## Tests

```bash
uv run --extra dev pytest        # or, with pip:  pip install -e ".[dev]" && pytest
```
