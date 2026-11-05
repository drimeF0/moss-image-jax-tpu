from dataclasses import dataclass


@dataclass(frozen=True)
class MossConfig:
    dim: int = 3072
    ffn_dim: int = 8192
    num_heads: int = 24
    num_layers: int = 42
    text_dim: int = 4096
    freq_dim: int = 256
    latent_channels: int = 128
    rope_base: float = 10000.0

    @property
    def head_dim(self) -> int:
        return self.dim // self.num_heads


MOSS_IMAGE_CONFIG = MossConfig()
