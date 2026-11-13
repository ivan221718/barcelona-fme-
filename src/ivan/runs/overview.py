"""Run page on the IVAN dashboard: loss and accuracy curves plus the final numbers."""
import plotly.express as px
import ivan

run = ivan.results.run()
steps = run.steps.frame()
run.figure(px.line(steps, x="_step", y=["train_loss", "val_loss"]), "losses", title="Loss per epoch")
run.figure(px.line(steps, x="_step", y="val_acc"), "val_acc", title="Validation accuracy per epoch")
run.table("summary", [run.summary], title="Final scores")
