"""Model loading helpers shared by SPA's SFT and Colab diagnostics."""


def text_config(config):
    """Return the language config for text-only or multimodal checkpoints."""
    return getattr(config, 'text_config', config)


def model_class(config):
    # Keep the official architecture/weight names even when inputs are text-only.
    # Dropping vision weights or rewriting the config would break checkpoint reuse.
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    if getattr(config, 'vision_config', None) is not None:
        return AutoModelForImageTextToText
    return AutoModelForCausalLM
