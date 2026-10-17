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
NEW_TOKENS = ["<class_Wall>", "<class_Door>", "<class_Window>", "<class_Room>"]

def rows_to_markdown(objects):
    # Target format Parse postprocessing expects: class + bbox + text lines
    lines = []
    for o in objects:
        x0, y0, x1, y1 = o["bbox"]
        lines.append(f'{o["class"]} [{x0},{y0},{x1},{y1}]: {o["text"]}')
    return "\n".join(lines)

class ParseDS(torch.utils.data.Dataset):
    def __init__(self, path, processor, tokenizer, max_len=2048):
        self.rows = [json.loads(l) for l in open(path)]
        self.proc, self.tok, self.max_len = processor, tokenizer, max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        img = Image.open(r["image"]).convert("RGB")
        enc = self.proc(images=[img], text=r["prompt"], return_tensors="pt", add_special_tokens=False)
        tgt = self.tok(rows_to_markdown(r["objects"]), return_tensors="pt",
                       max_length=self.max_len, truncation=True).input_ids[0]
        item = {k: v[0] for k, v in enc.items()}
        item["labels"] = tgt
        return item

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default="./ckpt")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--grad-ckpt", type=int, default=1)
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
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    ds = ParseDS(a.train, proc, tok)
    args = TrainingArguments(a.out, per_device_train_batch_size=a.batch,
                             gradient_accumulation_steps=8, num_train_epochs=a.epochs,
                             learning_rate=a.lr, bf16=True, logging_steps=10,
                             save_steps=200, save_total_limit=2, remove_unused_columns=False)
    Trainer(model, args, train_dataset=ds).train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    print(f"saved -> {a.out}")

if __name__ == "__main__":
    main()
