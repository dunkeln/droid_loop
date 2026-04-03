import base64
import io
import json
import random
from pathlib import Path

import anthropic
from PIL import Image

from .catalog import load_moments
from .catalog import REPO_ROOT

DEFAULT_DESCRIPTIONS = REPO_ROOT / "cluster_descriptions.json"

MODEL_ID = "claude-haiku-4-5-20251001"
LABEL_SAMPLE_SEED = 17


def _encode_image(image: Image.Image) -> str:
    """Encode a PIL image as base64 JPEG for the Anthropic API."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.standard_b64encode(buf.getvalue()).decode()


def _group_by_cluster(catalog_path: Path) -> dict[int, dict[int, list[int]]]:
    """Read the catalog and return {episode_id: {cluster_id: [frame_indices]}}."""
    result: dict[int, dict[int, list[int]]] = {}
    for moment in load_moments(catalog_path):
        ep = moment["episode_id"]
        cl = moment.get("cluster_id", -1)
        fi = moment["frame_index"]
        result.setdefault(ep, {}).setdefault(cl, []).append(fi)
    return result


def _as_pil_image(value: object) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    return Image.fromarray(value)


def describe_cluster(images: list[Image.Image], client: anthropic.Anthropic) -> str:
    """Send representative cluster frames to the VLM and return a short description."""
    content: list[dict] = []
    for img in images:
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": _encode_image(img),
            },
        })
    content.append({
        "type": "text",
        "text": (
            "These frames come from a robot arm manipulation task. "
            "They were grouped together because they are visually similar. "
            "In 5 words or fewer, describe what the robot is doing or what state it is in. "
            "Be specific — mention the gripper, object, or arm position if visible."
        ),
    })

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=50,
        messages=[{"role": "user", "content": content}],
    )
    return response.content[0].text.strip()


def label_clusters(
    catalog_path: Path,
    dataset_id: str,
    image_key: str,
    output_path: Path = DEFAULT_DESCRIPTIONS,
    n_samples: int = 3,
) -> dict:
    """Stream episodes, sample representative frames per cluster, label with VLM.

    The VLM never touches the scan pipeline — it queries clusters from the catalog
    as a retriever, decoupled from scoring entirely.

    Returns {episode_id: {cluster_id: description}} and writes to output_path.
    """
    from . import loader

    client = anthropic.Anthropic()
    rng = random.Random(LABEL_SAMPLE_SEED)
    groups = _group_by_cluster(catalog_path)
    target_episodes = set(groups.keys())
    descriptions: dict[str, dict[str, str]] = {}

    for episode in loader.stream_episodes(dataset_id):
        episode_id = episode[0]["episode_index"]
        if episode_id not in target_episodes:
            continue

        frame_lookup = {frame["frame_index"]: frame for frame in episode}
        ep_groups = groups[episode_id]
        ep_descs: dict[str, str] = {}

        for cluster_id, frame_indices in ep_groups.items():
            sampled = rng.sample(sorted(frame_indices), min(n_samples, len(frame_indices)))
            resolved_key = image_key
            if sampled and sampled[0] in frame_lookup and resolved_key not in frame_lookup[sampled[0]]:
                resolved_key = {
                    "observation.images.wrist_image_left": "observation.images.wrist_left",
                    "observation.images.exterior_image_1_left": "observation.images.exterior_1_left",
                    "observation.images.exterior_image_2_left": "observation.images.exterior_2_left",
                }.get(resolved_key, resolved_key)
            images = [
                _as_pil_image(frame_lookup[fi][resolved_key])
                for fi in sampled
                if fi in frame_lookup and resolved_key in frame_lookup[fi]
            ]
            if not images:
                continue
            ep_descs[str(cluster_id)] = describe_cluster(images, client)

        descriptions[str(episode_id)] = ep_descs

        if set(descriptions.keys()) >= {str(ep) for ep in target_episodes}:
            break

    output_path.write_text(json.dumps(descriptions, indent=2))
    return descriptions
