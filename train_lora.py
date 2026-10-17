"""LoRA fine-tune NVIDIA-Nemotron-Parse-2.0 on blueprint JSONL.
Brev: 1x A100 40GB, bf16. 3-5 epochs, stop at dim accuracy >85%.

  python train_lora.py --train ./data/train.jsonl --out ./ckpt --epochs 4
"""
import argparse, json
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

MODEL_ID = "nvidia/NVIDIA-Nemotron-Parse-2.0"
from classes import CLASS_TOKENS as NEW_TOKENS  # order matters: matches ckpt

def _quant_tables(tok):
    """index -> token string for the native coordinate tokens.

    The model emits normalized coord TOKENS, not bracketed pixel numbers:
    <x_0.3711><y_0.3539><x_0.625><y_0.6523><class_Picture>
    x has 1024 bins (k/1024), y has 1280 bins (k/1280). Building the target in
    this vocabulary is what makes earlier runs learnable at all -- the old
    "Wall [0,0,271,250]: text" target was a format the model never produces.
    """
    vx, vy = {}, {}
    for k in tok.get_vocab():
        if k.startswith("<x_"):
            vx[int(round(float(k[3:-1]) * 1024))] = k
        elif k.startswith("<y_"):
            vy[int(round(float(k[3:-1]) * 1280))] = k
    return vx, vy


def _qx(v, size, vx):
    k = max(0, min(1023, int(round(float(v) / size * 1024))))
    return vx.get(k, "<x_0.5>")


def _qy(v, size, vy):
    k = max(0, min(1279, int(round(float(v) / size * 1280))))
    return vy.get(k, "<y_0.5>")


def rows_to_markdown(objects, w, h, vx, vy):
    # Native output format: four coord tokens then the class token, then text.
    lines = []
    for o in objects:
        x0, y0, x1, y1 = o["bbox"]
        cls = o.get("class", "Room")
        text = (o.get("text") or "").strip()
        line = f"{_qx(x0, w, vx)}{_qy(y0, h, vy)}{_qx(x1, w, vx)}{_qy(y1, h, vy)}<class_{cls}>"
        if text:
            line += f" {text}"
        lines.append(line)
    return "\n".join(lines)

class ParseDS(torch.utils.data.Dataset):
    def __init__(self, path, processor, tokenizer, max_len=2048):
        self.rows = [json.loads(l) for l in open(path)]
        self.proc, self.tok, self.max_len = processor, tokenizer, max_len
        self.vx, self.vy = _quant_tables(tokenizer)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["image"]).convert("RGB")
        w, h = img.size
        # train text output too (room dims live there); data jsonl was built with
        # <predict_no_text_in_pic>, which contradicts a text-bearing target
        prompt = r["prompt"].replace("<predict_no_text_in_pic>", "<predict_text_in_pic>")
        enc = self.proc(images=[img], text=prompt, return_tensors="pt", add_special_tokens=False)
        tgt = self.tok(rows_to_markdown(r["objects"], w, h, self.vx, self.vy), return_tensors="pt",
                       max_length=self.max_len, truncation=True).input_ids[0]
        item = {k: v[0] for k, v in enc.items()}
        item["labels"] = tgt
        return item

def make_collate(pad_id):
    from torch.nn.utils.rnn import pad_sequence
    def collate(batch):
        out = {}
        for k in batch[0].keys():
            vals = [b[k] if torch.is_tensor(b[k]) else torch.tensor(b[k]) for b in batch]
            if k in ("input_ids", "attention_mask", "labels"):
                pv = pad_id if k in ("input_ids", "labels") else 0
                out[k] = pad_sequence(vals, batch_first=True, padding_value=pv)
            else:
                out[k] = torch.stack(vals)
        out["labels"][out["labels"] == pad_id] = -100
        return out
    return collate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default="./ckpt")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-ckpt", type=int, default=1)
    ap.add_argument("--full", type=int, default=0, help="1: full fine-tune, skip PEFT (needs 40GB+ VRAM)")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok.add_tokens(NEW_TOKENS)
    model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True,
                                      torch_dtype=torch.bfloat16).cuda().train()
    if a.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.resize_token_embeddings(len(tok))
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="SEQ_2_SEQ_LM")
    if not a.full:
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    ds = ParseDS(a.train, proc, tok)
    args = TrainingArguments(a.out, per_device_train_batch_size=a.batch,
                             gradient_accumulation_steps=8, num_train_epochs=a.epochs,
                             learning_rate=a.lr, bf16=True, logging_steps=10,
                             save_steps=200, save_total_limit=2, remove_unused_columns=False)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    Trainer(model, args, train_dataset=ds, data_collator=make_collate(pad_id)).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"saved -> {a.out}")

if __name__ == "__main__":
    main()
