# DROID Loop
DROID Loop scans robot manipulation video, surfaces rare failures, and turns them into reusable training signal.
<!-- demo.gif -->
It is built for multiview DROID episodes, async scoring, human review, and query-time VLM inspection.
The scan path loads episode media, scores frames, merges anomalies into incident windows, and catalogs them in SQLite.
The review path restores episodes from the DB, plays synchronized camera views, and overlays score traces on video.
The labeling path uses VLM descriptions only after mining, so semantic cost stays bounded.
The export path converts validated incidents into flywheel-ready signal artifacts.
State is durable across graceful restarts through `catalog.db` plus retained review media in `frames/`.
Run the API with `uv run uvicorn server:app --host 127.0.0.1 --port 8000` and the UI from `ui/`.
