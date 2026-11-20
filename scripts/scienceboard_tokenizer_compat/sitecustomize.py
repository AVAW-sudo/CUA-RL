"""Process-scoped schema adapter for the released ScienceBoard SFT tokenizer.

The checkpoint exports extra_special_tokens as a list. Transformers 4.x expects
that list under additional_special_tokens and reserves extra_special_tokens for
a named-token dictionary. Preserve token strings, IDs and checkpoint files.
"""
import os

if os.environ.get('CUA_SCI_TOKENIZER_MODEL_ROOT'):
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    _original_init = PreTrainedTokenizerBase.__init__
    _model_root = os.environ['CUA_SCI_TOKENIZER_MODEL_ROOT']

    def _compatible_init(self, *args, **kwargs):
        extra = kwargs.get('extra_special_tokens')
        if isinstance(extra, list) and kwargs.get('name_or_path') == _model_root:
            if not all(isinstance(token, str) for token in extra):
                raise TypeError('Unexpected non-string token in the pinned ScienceBoard tokenizer')
            kwargs = dict(kwargs)
            existing = list(kwargs.get('additional_special_tokens') or [])
            kwargs['additional_special_tokens'] = existing + [token for token in extra if token not in existing]
            kwargs['extra_special_tokens'] = {}
        return _original_init(self, *args, **kwargs)

    PreTrainedTokenizerBase.__init__ = _compatible_init
