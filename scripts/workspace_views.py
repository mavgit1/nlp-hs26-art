"""Create the W&B saved workspace views for Project 1 and print their URLs.

final view: only runs tagged final/baseline visible; all runs listed by val loss.
sweeps view: every sweep run visible, to compare hyperparameters.
"""
import os

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY = os.environ.get("WANDB_ENTITY", "marvin-huber-hochschule-luzern")
PROJECT = os.environ.get("WANDB_PROJECT", "nlp-hs26-art-p1")

runs = [r for r in wandb.Api().runs(f"{ENTITY}/{PROJECT}") if r.state == "finished"]


def curves(title: str, train: bool) -> ws.Section:
    # train is logged every 50 steps and noisy; validation once per epoch, unsmoothed
    smooth = {"smoothing_type": "exponential", "smoothing_factor": 0.6} if train else {"smoothing_type": "none"}
    split = "train" if train else "val"
    return ws.Section(name=title, is_open=True, layout_settings=ws.SectionLayoutSettings(columns=2, rows=1), panels=[
        wr.LinePlot(title=f"{split} loss", x="epoch", y=[f"{split}/loss"], **smooth),
        wr.LinePlot(title=f"{split} accuracy", x="epoch", y=[f"{split}/acc"], **smooth)])


def runset(visible) -> ws.RunsetSettings:
    return ws.RunsetSettings(
        order=[ws.Ordering(ws.Summary("best/val/loss"), ascending=True)],
        pinned_columns=["run:displayName", "summary:best/val/loss", "summary:best/val/acc", "summary:best/epoch"],
        run_settings={r.id: ws.RunSettings(disabled=not visible(r)) for r in runs})


settings = ws.WorkspaceSettings(x_axis="epoch", max_runs=20, tooltip_number_of_runs="default")

final = ws.Workspace(
    entity=ENTITY, project=PROJECT, name="Project 1: final models",
    sections=[curves("Validation", train=False), curves("Training", train=True),
              ws.Section(name="Final evaluation", is_open=True, panels=[
                  wr.BarPlot(title="validation vs test accuracy", metrics=["validation_eval/acc", "test_eval/acc"], orientation="h")])],
    settings=settings, runset_settings=runset(lambda r: bool({"final", "baseline"} & set(r.tags))))

sweeps = ws.Workspace(
    entity=ENTITY, project=PROJECT, name="Project 1: sweeps",
    sections=[ws.Section(name="Learning rate", is_open=True, panels=[
                  wr.ScatterPlot(title="lr vs best val loss", x="lr", y="best/val/loss", log_x=True),
                  wr.ParallelCoordinatesPlot(title="hyperparameters", columns=[
                      wr.ParallelCoordinatesPlotColumn(metric=wr.Config(c)) for c in
                      ("model", "format", "interactions", "cell", "hidden", "lr")] + [
                      wr.ParallelCoordinatesPlotColumn(metric=wr.SummaryMetric("best/val/loss"))])]),
              curves("Validation", train=False), curves("Training", train=True)],
    settings=settings, runset_settings=runset(lambda r: "sweep" in r.tags))

for view in (final, sweeps):
    print(view.name, view.save().url)
