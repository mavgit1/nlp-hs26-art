"""Create (or update in place) the W&B saved workspace views for Project 1.

Re-run after new runs: pass the view URLs to keep the links stable.

final view: only runs tagged final/baseline, grouped by model (mean ± std over seeds).
sweeps view: every sweep run visible, to compare hyperparameters.
"""
import argparse
import os

import wandb
import wandb_workspaces.reports.v2 as wr
import wandb_workspaces.workspaces as ws

ENTITY = os.environ.get("WANDB_ENTITY", "marvin-huber-hochschule-luzern")
PROJECT = os.environ.get("WANDB_PROJECT", "nlp-hs26-art-p1")

p = argparse.ArgumentParser()
p.add_argument("--final-url", help="existing final view to update")
p.add_argument("--sweeps-url", help="existing sweeps view to update")
args = p.parse_args()

runs = [r for r in wandb.Api().runs(f"{ENTITY}/{PROJECT}") if r.state == "finished"]


SEED_BAND = {"groupby_aggfunc": "mean", "groupby_rangefunc": "stddev"}  # mean curve, ±std over seeds


def curves(title: str, train: bool, grouped: bool = False) -> ws.Section:
    # train is logged every 50 steps and noisy; validation once per epoch, unsmoothed
    opts = {"smoothing_type": "exponential", "smoothing_factor": 0.6} if train else {"smoothing_type": "none"}
    opts |= SEED_BAND if grouped else {}
    split = "train" if train else "val"
    return ws.Section(name=title, is_open=True, layout_settings=ws.SectionLayoutSettings(columns=2, rows=1), panels=[
        wr.LinePlot(title=f"{split} loss", x="epoch", y=[f"{split}/loss"], **opts),
        wr.LinePlot(title=f"{split} accuracy", x="epoch", y=[f"{split}/acc"], **opts)])


def runset(visible, groupby=(), filters="") -> ws.RunsetSettings:
    return ws.RunsetSettings(
        groupby=list(groupby), filters=filters,
        order=[ws.Ordering(ws.Summary("best/val/loss"), ascending=True)],
        pinned_columns=["run:displayName", "summary:best/val/loss", "summary:best/val/acc", "summary:best/epoch"],
        run_settings={r.id: ws.RunSettings(disabled=not visible(r)) for r in runs})


settings = ws.WorkspaceSettings(x_axis="epoch", max_runs=20, tooltip_number_of_runs="default")

final = ws.Workspace(
    entity=ENTITY, project=PROJECT, name="Project 1: final models",
    sections=[curves("Validation", train=False, grouped=True), curves("Training", train=True, grouped=True),
              ws.Section(name="Final evaluation", is_open=True, panels=[
                  wr.BarPlot(title="validation vs test accuracy", metrics=["validation_eval/acc", "test_eval/acc"],
                             orientation="h", **SEED_BAND)])],
    settings=settings,
    # one group per model: 3 seeds each
    runset_settings=runset(lambda r: bool({"final", "baseline"} & set(r.tags)),
                           groupby=[ws.Config("model"), ws.Config("hyp_only")],
                           filters=[ws.Tags().isin(["final", "baseline"])]))

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

for view, url in ((final, args.final_url), (sweeps, args.sweeps_url)):
    if url:  # keep the saved view's identity, replace its contents
        old = ws.Workspace.from_url(url)
        old.name, old.sections, old.settings, old.runset_settings = (
            view.name, view.sections, view.settings, view.runset_settings)
        view = old
    url = view.save().url  # forge serves the app under /wandb/
    print(view.name, url.replace("forge.coreweave.com/", "forge.coreweave.com/wandb/", 1) if "/wandb/" not in url else url)
