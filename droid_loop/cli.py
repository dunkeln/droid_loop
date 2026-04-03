from pathlib import Path

import typer
from PIL import Image
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from . import catalog, labeler, loader, signal
from .clip_extractor import ClipRequest, extract_clip
from .scorer import FrameScorer

app = typer.Typer(help="DROID Loop — surface rare moments and build training signal.")
console = Console()


def _as_pil_image(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    return Image.fromarray(value)


@app.command()
def stream(dataset_id: str = loader.DROID_DATASET) -> None:
    """Stream episodes and print episode/frame counts."""
    for episode in loader.stream_episodes(dataset_id):
        typer.echo(f"episode={episode[0]['episode_index']}  frames={len(episode)}")


@app.command()
def scan(
    dataset_id: str = loader.DROID_DATASET,
    contamination: float = 0.01,
    image_key: str = "observation.images.wrist_left",
    catalog_path: Path = catalog.DB_PATH,
    context_window: int = 4,
    start_episode: int = 0,
    max_episodes: int | None = None,
    max_frames_per_episode: int = 300,
    embed_batch_size: int = 32,
) -> None:
    """Score every episode with SigLIP and log anomalous frames to the catalog.

    Each saved moment includes cluster metadata (size, span) and a temporal
    context window of surrounding frame indices.
    """
    scorer = FrameScorer(contamination=contamination, batch_size=embed_batch_size)
    episode_fps = loader.dataset_fps(dataset_id)
    total_flagged = 0
    max_frames = max(1, max_frames_per_episode)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        TaskProgressColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)

        for episode in loader.stream_episodes(
            dataset_id,
            image_keys=[image_key],
            start_episode=start_episode,
            max_episodes=max_episodes,
        ):
            episode_id = episode[0]["episode_index"]
            progress.update(task, description=f"episode={episode_id}  scoring {len(episode)} frames...")

            # Resolve image key — not all episodes have every camera
            resolved_key = next(
                (k for k in [image_key,
                              "observation.images.exterior_1_left",
                              "observation.images.exterior_2_left",
                              "observation.images.wrist_left",
                              # Legacy aliases for older datasets.
                              "observation.images.exterior_image_1_left",
                              "observation.images.exterior_image_2_left",
                              "observation.images.wrist_image_left",
                              "observation.images.wrist_image_right"]
                 if episode[0].get(k) is not None),
                None,
            )
            if resolved_key is None:
                console.print(f"[yellow]episode={episode_id} skipped — no image key found[/yellow]")
                continue

            camera_keys = sorted(
                {
                    key
                    for frame in episode
                    for key, value in frame.items()
                    if key.startswith("observation.images.") and value is not None
                }
            )
            if not camera_keys:
                continue
            if resolved_key not in camera_keys:
                resolved_key = camera_keys[0]

            valid = [f for f in episode if any(f.get(k) is not None for k in camera_keys)]
            if not valid:
                continue
            if len(valid) > max_frames:
                step = max(1, len(valid) // max_frames)
                valid = valid[::step][:max_frames]
                console.print(
                    f"[yellow]episode={episode_id} downsampled "
                    f"{len(episode)} -> {len(valid)} frames (step={step})[/yellow]"
                )

            frame_indices = [frame["frame_index"] for frame in valid]
            episode = valid
            images_by_camera = {
                key: [
                    _as_pil_image(frame[key]).convert("RGB") if frame.get(key) is not None else None
                    for frame in episode
                ]
                for key in camera_keys
            }
            bad_indices, cluster_labels, cluster_meta, _, _ = scorer.flag_multiview(
                images_by_camera=images_by_camera,
                frame_indices=frame_indices,
                primary_camera=resolved_key,
            )

            for idx in bad_indices:
                frame = episode[idx]
                cl_id = int(cluster_labels[idx])
                meta = cluster_meta.get(cl_id, {})

                # Temporal context: ±context_window frames, clamped to episode bounds
                lo = max(0, idx - context_window)
                hi = min(len(episode) - 1, idx + context_window)
                context = [
                    episode[j]["frame_index"]
                    for j in range(lo, hi + 1)
                    if j != idx
                ]
                clip, _ = extract_clip(
                    episode_rows=episode,
                    request=ClipRequest(
                        anchor_frame_index=int(frame["frame_index"]),
                        pre_s=2.0,
                        post_s=2.0,
                        max_frames=16,
                        strategy="uniform",
                    ),
                    fps=episode_fps,
                )

                catalog.save_moment(
                    {
                        "episode_id": episode_id,
                        "frame_index": frame["frame_index"],
                        "label": "anomaly",
                        "cluster_id": cl_id,
                        "cluster_size": meta.get("size", 0),
                        "cluster_span": meta.get("span", []),
                        "context_window": context,
                        "clip_pre_s": 2.0,
                        "clip_post_s": 2.0,
                        "clip_fps": episode_fps,
                        "clip_strategy": "uniform",
                        "clip_frame_indices": [f.frame_index for f in clip.frames],
                        "clip_anchor_timestamp_s": clip.anchor_timestamp_s,
                        "clip_start_timestamp_s": clip.start_timestamp_s,
                        "clip_end_timestamp_s": clip.end_timestamp_s,
                    },
                    catalog_path,
                )

            total_flagged += len(bad_indices)
            progress.update(task, description=f"episode={episode_id}  flagged={len(bad_indices)}/{len(episode)}  total={total_flagged}")

    console.print(f"[green]Done.[/green] {total_flagged} moments logged to {catalog_path}")


@app.command()
def log(
    episode_id: int,
    frame_index: int,
    label: str,
    catalog_path: Path = catalog.DB_PATH,
) -> None:
    """Log a surfaced bad moment to the catalog."""
    moment = {"episode_id": episode_id, "frame_index": frame_index, "label": label}
    catalog.save_moment(moment, catalog_path)
    typer.echo(f"Logged: {moment}")


@app.command()
def label(
    catalog_path: Path = catalog.DB_PATH,
    dataset_id: str = loader.DROID_DATASET,
    image_key: str = "observation.images.wrist_left",
    output_path: Path = Path("cluster_descriptions.json"),
    n_samples: int = 3,
) -> None:
    """Query the VLM to generate semantic descriptions for each cluster."""
    console.print("[cyan]Querying VLM for cluster descriptions...[/cyan]")
    descriptions = labeler.label_clusters(catalog_path, dataset_id, image_key, output_path, n_samples)
    total = sum(len(v) for v in descriptions.values())
    console.print(f"[green]Done.[/green] {total} clusters labeled → {output_path}")


@app.command()
def export(
    catalog_path: Path = catalog.DB_PATH,
    output_path: Path = Path("training_signal.json"),
) -> None:
    """Export cataloged moments as a training signal file."""
    signal.export(catalog_path, output_path)
