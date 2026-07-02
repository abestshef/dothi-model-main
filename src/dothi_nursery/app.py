"""Shiny for Python app: Dothistroma spread & control in a forest nursery.

Reproduces the interactive notebook (``Dothi_model_for_RSE.ipynb``) as a
standalone web app. Choose parameters in the sidebar and click *Run the model*;
the two time-course plots and the 4x3 field snapshots are drawn below.

Run it with either::

    dothi-nursery                     # console script installed with the package
    shiny run --reload src/dothi_nursery/app.py:app
"""

from __future__ import annotations

import contextlib

import numpy as np
from shiny import App, reactive, render, ui

# Absolute (not relative) imports so this module works both when imported as
# part of the package (the `dothi-nursery` console script) and when Shiny loads
# it directly by file path (`shiny run src/dothi_nursery/app.py:app`).
from dothi_nursery.model import SimParams, run_simulation
from dothi_nursery.plotting import placeholder, plot_snapshots, plot_timecourse

# Intro text lifted from the notebook's opening markdown cell.
INTRO = """
### A model of *Dothistroma* spread and control in a forest nursery

Explore the spread of *Dothistroma septosporum* through a forest nursery and the
effect of different control measures.

Saplings sit on a 0.5&nbsp;m &times; 0.5&nbsp;m grid with every 5th row left empty.
Along the southern edge there is an external source of infection (e.g. unmanaged
mature woodland). Infection disperses plant-to-plant with distance (95% of
infections within 2&nbsp;m), and is **seasonal** &mdash; no spread in winter.
Newly infected plants are asymptomatic (and non-infectious) for an incubation
period before showing symptoms.

At a month of your choosing the field is surveyed and **four control strategies**
are compared:

1. Remove only symptomatic plants and **replace** them with healthy saplings
2. Remove only symptomatic plants and **leave the sites empty**
3. Remove symptomatic plants **and those within a radius**, and replace
4. Remove symptomatic plants **and those within a radius**, and leave empty

The model is **stochastic**, so the outcome differs on every run.
"""

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_radio_buttons(
            "beta", "Within-field infection rate",
            ["Low", "Moderate", "High"], selected="Moderate",
        ),
        ui.input_radio_buttons(
            "betap", "External infection rate",
            ["Low", "Moderate", "High"], selected="Moderate",
        ),
        ui.input_radio_buttons(
            "gamma", "Average incubation period",
            ["2 months", "4 months", "6 months"], selected="4 months",
        ),
        ui.input_slider("control", "Control month", min=6, max=24, value=16, step=1),
        ui.input_slider("radius", "Control radius (m)", min=0.5, max=5, value=2, step=0.5),
        ui.input_action_button("run", "Run the model", class_="btn-primary"),
        ui.markdown(
            "_The model is stochastic and can take a few seconds to run._"
        ),
        width=330,
        title="Parameters",
    ),
    ui.markdown(INTRO),
    ui.h4("Time courses"),
    ui.output_plot("timecourse", height="430px"),
    ui.h4("Field snapshots"),
    ui.output_plot("snapshots", height="950px"),
    title="Dothistroma nursery model",
    fillable=False,
)


def server(input, output, session):

    @reactive.calc
    @reactive.event(input.run)
    def sim():
        params = SimParams.from_choices(
            input.beta(), input.betap(), input.gamma(),
            input.control(), input.radius(),
        )
        # A progress bar is nice UX, but must never be able to break a run, so
        # fall back to no progress reporting if the session can't provide one.
        try:
            prog_cm = ui.Progress(min=0, max=5)
        except Exception:
            prog_cm = contextlib.nullcontext(None)
        with prog_cm as prog:
            def report(stage, message):
                if prog is not None:
                    prog.set(stage, message=message)
            return run_simulation(params, rng=np.random.default_rng(), progress=report)

    @render.plot
    def timecourse():
        if not input.run():
            return placeholder("Choose parameters, then click 'Run the model'.")
        return plot_timecourse(sim())

    @render.plot
    def snapshots():
        if not input.run():
            return placeholder("The field snapshots will appear here after a run.")
        return plot_snapshots(sim())


app = App(app_ui, server)
