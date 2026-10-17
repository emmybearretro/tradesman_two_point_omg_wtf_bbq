"""Patch NVIDIA-Nemotron-Parse-2.0 remote code for transformers >= 5.x.

transformers 5 drops `inputs_embeds` from the decoder's output object, so the
model's own forward() crashes in the training branch:

    decoder_inputs_embeds = decoder_outputs.inputs_embeds
    AttributeError: 'BaseModelOutputWithPastAndCrossAttentions' object has no
                    attribute 'inputs_embeds'

The value is only used to build the shifted embeddings fed to the extra heads,
so recomputing it from `decoder_input_ids` via `self.decoder.embed_tokens`
(scaled embedding, same weights) is exactly equivalent. Idempotent; safe to run
on every setup.

  python3 fix_remote_code.py
"""
import glob
import os
import sys

OLD = "            decoder_inputs_embeds = decoder_outputs.inputs_embeds\n"
NEW = (
    "            # patched for transformers>=5: decoder output no longer carries inputs_embeds\n"
    "            decoder_inputs_embeds = getattr(decoder_outputs, \"inputs_embeds\", None)\n"
    "            if decoder_inputs_embeds is None:\n"
    "                decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids)\n"
)

PATTERN = os.path.expanduser(
    "~/.cache/huggingface/modules/transformers_modules/nvidia/"
    "*/**/hf_nemotron_parse_modeling.py"
)


def main():
    files = sorted(glob.glob(PATTERN, recursive=True))
    if not files:
        print("no cached remote code found (model not downloaded yet?)", file=sys.stderr)
        return 1
    patched = 0
    for f in files:
        src = open(f).read()
        if NEW in src:
            print(f"already patched: {f}")
            continue
        if OLD not in src:
            print(f"anchor line not found, skipped: {f}")
            continue
        open(f, "w").write(src.replace(OLD, NEW))
        print(f"patched: {f}")
        patched += 1
    print(f"done ({patched} file(s) patched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
