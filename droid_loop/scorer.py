import numpy as np
import torch
from PIL import Image
from sklearn.cluster import HDBSCAN
from sklearn.ensemble import IsolationForest
from transformers import AutoModel, AutoProcessor

MODEL_ID = "google/siglip-base-patch16-224"


class FrameScorer:
    def __init__(
        self,
        model_id: str = MODEL_ID,
        contamination: float = 0.01,
        batch_size: int = 32,
        temporal_jump_weight: float = 0.45,
        jump_flag_quantile: float = 0.92,
    ):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        self.detector = IsolationForest(contamination=contamination, random_state=42)
        self.batch_size = max(1, batch_size)
        self.temporal_jump_weight = float(min(0.95, max(0.0, temporal_jump_weight)))
        self.jump_flag_quantile = float(min(0.999, max(0.5, jump_flag_quantile)))

    def embed(self, images: list[Image.Image]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for i in range(0, len(images), self.batch_size):
                batch = images[i : i + self.batch_size]
                inputs = self.processor(images=batch, return_tensors="pt")
                raw = self.model.get_image_features(**inputs)
                features = self._as_feature_tensor(raw)
                chunks.append(features.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def _as_feature_tensor(self, value: object) -> torch.Tensor:
        """Normalize HF model outputs to a feature tensor."""
        if isinstance(value, torch.Tensor):
            return value
        if hasattr(value, "pooler_output"):
            pooled = getattr(value, "pooler_output")
            if isinstance(pooled, torch.Tensor):
                return pooled
        if hasattr(value, "last_hidden_state"):
            hidden = getattr(value, "last_hidden_state")
            if isinstance(hidden, torch.Tensor):
                # CLS-style embedding fallback.
                return hidden[:, 0, :]
        raise TypeError(f"Unsupported image feature output type: {type(value).__name__}")

    def cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """Cluster embeddings into semantic groups. Returns cluster labels (-1 = noise/anomaly)."""
        min_size = max(5, len(embeddings) // 20)
        # Pin copy behavior to avoid sklearn FutureWarning and keep current semantics.
        return HDBSCAN(min_cluster_size=min_size, copy=False).fit_predict(embeddings)

    def _cluster_meta(
        self, cluster_labels: np.ndarray, frame_indices: list[int]
    ) -> dict[int, dict]:
        """Compute per-cluster metadata: size and frame span.

        Returns {cluster_id: {"size": int, "span": [min_frame, max_frame]}}
        """
        meta: dict[int, dict] = {}
        for pos, cl in enumerate(cluster_labels):
            cl_id = int(cl)
            fi = frame_indices[pos]
            if cl_id not in meta:
                meta[cl_id] = {"size": 0, "span": [fi, fi]}
            meta[cl_id]["size"] += 1
            meta[cl_id]["span"][0] = min(meta[cl_id]["span"][0], fi)
            meta[cl_id]["span"][1] = max(meta[cl_id]["span"][1], fi)
        return meta

    def flag(
        self,
        images: list[Image.Image],
        frame_indices: list[int] | None = None,
    ) -> tuple[list[int], np.ndarray, dict[int, dict], list[float]]:
        """Return (anomalous_indices, cluster_labels, cluster_meta, anomaly_scores).

        A frame is flagged if Isolation Forest marks it as an outlier
        OR if HDBSCAN cannot assign it to any semantic cluster.

        cluster_meta keys are cluster IDs, values are {"size", "span"}.
        anomaly_scores is a 0..1 scaled anomaly intensity per frame.
        """
        if frame_indices is None:
            frame_indices = list(range(len(images)))

        embeddings = self.embed(images)
        return self._flag_from_embeddings(embeddings, frame_indices)

    def _flag_from_embeddings(
        self,
        embeddings: np.ndarray,
        frame_indices: list[int],
    ) -> tuple[list[int], np.ndarray, dict[int, dict], list[float]]:
        n = len(frame_indices)
        if n == 0:
            empty = np.zeros((0,), dtype=int)
            return [], empty, {}, []
        if n < 2:
            cluster_labels = np.zeros((n,), dtype=int)
            return [], cluster_labels, self._cluster_meta(cluster_labels, frame_indices), [0.0] * n

        iso_labels = self.detector.fit_predict(embeddings)
        # IsolationForest.decision_function: higher = more normal, lower = more anomalous.
        raw_scores = -self.detector.decision_function(embeddings)
        lo = float(np.min(raw_scores))
        hi = float(np.max(raw_scores))
        if hi <= lo:
            iso_scores = [0.0 for _ in raw_scores]
        else:
            iso_scores = [float((s - lo) / (hi - lo)) for s in raw_scores]
        jump_scores = self._temporal_jump_scores(embeddings)
        w = self.temporal_jump_weight
        anomaly_scores = [
            float((1.0 - w) * iso_scores[i] + w * jump_scores[i])
            for i in range(n)
        ]
        cluster_labels = self.cluster(embeddings)
        jump_threshold = float(np.quantile(jump_scores, self.jump_flag_quantile))

        flagged = [
            i for i, (iso, cl) in enumerate(zip(iso_labels, cluster_labels))
            if iso == -1 or cl == -1 or jump_scores[i] >= jump_threshold
        ]
        cluster_meta = self._cluster_meta(cluster_labels, frame_indices)
        return flagged, cluster_labels, cluster_meta, anomaly_scores

    def _temporal_jump_scores(self, embeddings: np.ndarray) -> np.ndarray:
        n = embeddings.shape[0]
        if n < 2:
            return np.zeros((n,), dtype=float)

        # Change magnitude between consecutive frames; project each jump to both
        # adjacent frames to avoid missing short transient failures.
        deltas = np.linalg.norm(embeddings[1:] - embeddings[:-1], axis=1)
        jump_raw = np.zeros((n,), dtype=float)
        jump_raw[1:] = np.maximum(jump_raw[1:], deltas)
        jump_raw[:-1] = np.maximum(jump_raw[:-1], deltas)
        lo = float(np.min(jump_raw))
        hi = float(np.max(jump_raw))
        if hi <= lo:
            return np.zeros((n,), dtype=float)
        return (jump_raw - lo) / (hi - lo)

    def flag_multiview(
        self,
        images_by_camera: dict[str, list[Image.Image | None]],
        frame_indices: list[int],
        primary_camera: str | None = None,
    ) -> tuple[list[int], np.ndarray, dict[int, dict], list[float], dict[str, dict[str, list]]]:
        """Fuse per-camera unsupervised scoring into one frame-level signal.

        Returns:
          (flagged_indices, fused_cluster_labels, fused_cluster_meta, fused_scores, per_camera)

        per_camera shape:
          {
            camera_key: {
              "scores": list[float],      # 0..1 per frame index (0 when camera frame missing)
              "clusters": list[int | None],
              "flagged": list[bool],
            }
          }
        """
        n = len(frame_indices)
        if n == 0:
            empty = np.zeros((0,), dtype=int)
            return [], empty, {}, [], {}
        if not images_by_camera:
            cluster_labels = np.zeros((n,), dtype=int)
            return [], cluster_labels, self._cluster_meta(cluster_labels, frame_indices), [0.0] * n, {}

        fused_scores = np.zeros((n,), dtype=float)
        fused_flags = np.zeros((n,), dtype=bool)
        per_camera: dict[str, dict[str, list]] = {}

        for camera_key, seq in images_by_camera.items():
            if len(seq) != n:
                raise ValueError(
                    f"camera '{camera_key}' frame count {len(seq)} does not match frame_indices {n}"
                )
            available_positions = [pos for pos, img in enumerate(seq) if img is not None]
            camera_scores = [0.0] * n
            camera_clusters: list[int | None] = [None] * n
            camera_flags = [False] * n
            if not available_positions:
                per_camera[camera_key] = {
                    "scores": camera_scores,
                    "clusters": camera_clusters,
                    "flagged": camera_flags,
                }
                continue

            camera_images = [seq[pos] for pos in available_positions]
            embeddings = self.embed([img for img in camera_images if img is not None])
            local_frame_indices = [frame_indices[pos] for pos in available_positions]
            local_flagged, local_clusters, _, local_scores = self._flag_from_embeddings(
                embeddings, local_frame_indices
            )
            local_flagged_set = set(local_flagged)
            for local_pos, global_pos in enumerate(available_positions):
                score = float(local_scores[local_pos])
                camera_scores[global_pos] = score
                camera_clusters[global_pos] = int(local_clusters[local_pos])
                is_flagged = local_pos in local_flagged_set
                camera_flags[global_pos] = is_flagged
                fused_scores[global_pos] = max(fused_scores[global_pos], score)
                fused_flags[global_pos] = fused_flags[global_pos] or is_flagged

            per_camera[camera_key] = {
                "scores": camera_scores,
                "clusters": camera_clusters,
                "flagged": camera_flags,
            }

        if primary_camera is None or primary_camera not in per_camera:
            primary_camera = next(iter(per_camera.keys()))
        primary_clusters = per_camera[primary_camera]["clusters"]

        fused_cluster_labels = np.full((n,), -1, dtype=int)
        for i in range(n):
            if primary_clusters[i] is not None:
                fused_cluster_labels[i] = int(primary_clusters[i])
                continue
            for camera in per_camera.values():
                cl = camera["clusters"][i]
                if cl is not None:
                    fused_cluster_labels[i] = int(cl)
                    break

        flagged = [i for i in range(n) if bool(fused_flags[i])]
        fused_cluster_meta = self._cluster_meta(fused_cluster_labels, frame_indices)
        return flagged, fused_cluster_labels, fused_cluster_meta, fused_scores.tolist(), per_camera
