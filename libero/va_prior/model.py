import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal


class TinyBackbone(nn.Module):
    def __init__(self, output_dim=384):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, output_dim, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )

    def forward(self, x):
        return self.net(x)


class FrozenHFBackbone(nn.Module):
    MODELS = {"dinov2": "facebook/dinov2-small", "siglip": "google/siglip-base-patch16-224"}

    def __init__(self, name):
        super().__init__()
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ImportError("Install requirements-va.txt for DINOv2/SigLIP") from exc
        self.model = AutoModel.from_pretrained(self.MODELS[name])
        processor = AutoImageProcessor.from_pretrained(self.MODELS[name])
        self.register_buffer("image_mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor(processor.image_std).view(1, 3, 1, 1))
        self.model.requires_grad_(False).eval()
        self.output_dim = self.model.config.hidden_size

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, x):
        x = F.interpolate(x, (224, 224), mode="bilinear", align_corners=False)
        x = (x - self.image_mean) / self.image_std
        with torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                return self.model.get_image_features(pixel_values=x)
            out = self.model(pixel_values=x)
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output
        return out.last_hidden_state[:, 0]


class Encoder(nn.Module):
    def __init__(self, backbone, proprio_dim, hidden_dim=256):
        super().__init__()
        if backbone == "tiny":
            self.vision, vision_dim = TinyBackbone(384), 384
        else:
            self.vision = FrozenHFBackbone(backbone)
            vision_dim = self.vision.output_dim
        self.camera_embed = nn.Linear(vision_dim, hidden_dim)
        self.proprio_embed = nn.Linear(proprio_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 4, batch_first=True)
        self.fusion = nn.TransformerEncoder(layer, 2)
        temporal_layer = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 4, batch_first=True)
        self.temporal = nn.TransformerEncoder(temporal_layer, 2)
        self.temporal_position = nn.Parameter(torch.zeros(1, 16, hidden_dim))

    def forward(self, images, proprio):
        if images.ndim == 5:
            images, proprio = images[:, None], proprio[:, None]
        b, time, cameras, c, h, w = images.shape
        v = self.vision(images.reshape(b * time * cameras, c, h, w)).reshape(b * time, cameras, -1)
        p = proprio.reshape(b * time, -1)
        tokens = torch.cat([self.camera_embed(v), self.proprio_embed(p).unsqueeze(1)], 1)
        per_time = self.fusion(tokens).mean(1).view(b, time, -1)
        if time > self.temporal_position.shape[1]:
            raise ValueError(f"Observation horizon {time} exceeds 16")
        return self.temporal(per_time + self.temporal_position[:, :time])[:, -1]


class FlowHead(nn.Module):
    def __init__(self, context_dim, output_dim, head_dim=512):
        super().__init__()
        self.output_dim = output_dim
        self.time = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        self.net = nn.Sequential(nn.Linear(context_dim + output_dim + 64, head_dim), nn.SiLU(),
                                 nn.Linear(head_dim, head_dim), nn.SiLU(),
                                 nn.Linear(head_dim, output_dim))

    def velocity(self, x, t, context):
        return self.net(torch.cat([x, context, self.time(t)], -1))

    def loss(self, target, context):
        noise = torch.randn_like(target)
        t = torch.rand(target.shape[0], 1, device=target.device)
        xt = (1 - t) * noise + t * target
        return F.mse_loss(self.velocity(xt, t, context), target - noise)

    def sample(self, context, n=8, steps=20):
        context = context.repeat_interleave(n, 0)
        x = torch.randn(context.shape[0], self.output_dim, device=context.device)
        dt = 1.0 / steps
        for i in range(steps):
            t = torch.full((len(x), 1), i * dt, device=x.device)
            x = x + dt * self.velocity(x, t, context)
        return x


class VAPriorModel(nn.Module):
    def __init__(self, head, backbone, proprio_dim, continuous_dim, horizon=10,
                 hidden_dim=256, num_modes=5, action_mean=None, action_std=None,
                 action_head_dim=512):
        super().__init__()
        self.head_type, self.horizon, self.continuous_dim = head, horizon, continuous_dim
        self.action_head_dim = action_head_dim
        self.encoder = Encoder(backbone, proprio_dim, hidden_dim)
        mean = torch.zeros(continuous_dim) if action_mean is None else torch.as_tensor(action_mean)
        std = torch.ones(continuous_dim) if action_std is None else torch.as_tensor(action_std)
        self.register_buffer("action_mean", mean.float())
        self.register_buffer("action_std", std.float())
        flat = horizon * continuous_dim
        self.gripper = nn.Sequential(nn.Linear(hidden_dim + flat, hidden_dim), nn.ReLU(),
                                     nn.Linear(hidden_dim, horizon))
        self.candidate_projector = nn.Sequential(nn.Linear(flat, hidden_dim), nn.Tanh())
        if head == "deterministic":
            self.head = nn.Sequential(nn.Linear(hidden_dim, action_head_dim), nn.ReLU(),
                                      nn.Linear(action_head_dim, flat))
        elif head == "gmm":
            self.num_modes = num_modes
            self.shared = nn.Sequential(nn.Linear(hidden_dim, action_head_dim), nn.ReLU())
            self.means = nn.Linear(action_head_dim, flat * num_modes)
            self.logstds = nn.Linear(action_head_dim, flat * num_modes)
            self.logits = nn.Linear(action_head_dim, num_modes)
        elif head == "flow":
            self.head = FlowHead(hidden_dim, flat, action_head_dim)
        else:
            raise ValueError(head)

    def action_head_parameter_count(self):
        """Count parameters in the continuous-action head only."""
        if self.head_type in ("deterministic", "flow"):
            parameters = self.head.parameters()
        else:
            modules = (self.shared, self.means, self.logstds, self.logits)
            parameters = (parameter for module in modules for parameter in module.parameters())
        return sum(parameter.numel() for parameter in parameters)

    def context(self, batch):
        return self.encoder(batch["images"], batch["proprio"])

    def _gmm(self, context):
        z, d = self.shared(context), self.horizon * self.continuous_dim
        means = self.means(z).view(-1, self.num_modes, d)
        stds = F.softplus(self.logstds(z).view(-1, self.num_modes, d)) + 1e-4
        return MixtureSameFamily(Categorical(logits=self.logits(z)), Independent(Normal(means, stds), 1))

    def loss(self, batch):
        context = self.context(batch)
        target = batch["continuous"].flatten(1)
        if self.head_type == "deterministic":
            action_loss = F.mse_loss(self.head(context), target)
        elif self.head_type == "gmm":
            action_loss = -self._gmm(context).log_prob(target).mean()
        else:
            action_loss = self.head.loss(target, context)
        grip_loss = F.binary_cross_entropy_with_logits(self.gripper(torch.cat([context, target], -1)), batch["gripper"])
        return action_loss + grip_loss, {"action": action_loss.detach(), "gripper": grip_loss.detach()}

    @torch.no_grad()
    def candidates(self, batch, k=8, flow_steps=20, cluster_threshold=1.0, cluster=True):
        """Return the stable interface consumed by a future language/MoE selector."""
        context = self.context(batch)
        b, d = len(context), self.horizon * self.continuous_dim
        if self.head_type == "deterministic":
            raw = self.head(context)[:, None]
        elif self.head_type == "gmm":
            raw = self._gmm(context).sample((k,)).permute(1, 0, 2)
        else:
            raw = self.head.sample(context, k, flow_steps).view(b, k, d)
        expanded_context = context[:, None].expand(-1, raw.shape[1], -1)
        grip_input = torch.cat([expanded_context, raw], -1).flatten(0, 1)
        grip = (self.gripper(grip_input).sigmoid() > 0.5).float().view(b, raw.shape[1], self.horizon)
        if cluster:
            medoids, weights, medoid_indices = cluster_medoids(raw, cluster_threshold)
        else:
            medoids = raw
            weights = torch.full((b, raw.shape[1]), 1.0 / raw.shape[1],
                                 device=raw.device, dtype=raw.dtype)
            medoid_indices = torch.arange(raw.shape[1], device=raw.device)[None].expand(b, -1)
        grip = grip.gather(1, medoid_indices[..., None].expand(-1, -1, self.horizon))
        features = self.candidate_projector(medoids) + context[:, None]
        continuous = medoids.view(b, medoids.shape[1], self.horizon, self.continuous_dim)
        continuous = continuous * self.action_std + self.action_mean
        full_chunks = torch.cat([continuous, grip[:, :medoids.shape[1], :, None] * 2 - 1], -1)
        return {
            "candidate_chunks": full_chunks,
            "prior_weights": weights,
            "candidate_features": features,
            "gripper_chunks": grip[:, :medoids.shape[1]],
        }


def cluster_medoids(samples, threshold):
    """Greedy radius clustering with padded medoids and empirical prior mass."""
    outputs, masses, all_indices = [], [], []
    max_k = samples.shape[1]
    for points in samples:
        remaining, clusters = list(range(len(points))), []
        while remaining:
            seed = remaining[0]
            distances = torch.linalg.vector_norm(points[remaining] - points[seed], dim=-1) / math.sqrt(points.shape[-1])
            members = [remaining[j] for j in torch.where(distances <= threshold)[0].tolist()]
            cluster = points[members]
            medoid = members[torch.cdist(cluster, cluster).sum(1).argmin().item()]
            clusters.append((points[medoid], len(members)))
            remaining = [x for x in remaining if x not in members]
        meds = torch.stack([x[0] for x in clusters])
        indices = torch.tensor([members_index(points, x[0]) for x in clusters], device=points.device)
        mass = torch.tensor([x[1] for x in clusters], device=points.device, dtype=points.dtype)
        pad = max_k - len(clusters)
        outputs.append(torch.cat([meds, meds[-1:].expand(pad, -1)]))
        masses.append(torch.cat([mass, torch.zeros(pad, device=points.device)]) / mass.sum())
        all_indices.append(torch.cat([indices, indices[-1:].expand(pad)]))
    return torch.stack(outputs), torch.stack(masses), torch.stack(all_indices)


def members_index(points, medoid):
    return torch.linalg.vector_norm(points - medoid, dim=-1).argmin().item()
