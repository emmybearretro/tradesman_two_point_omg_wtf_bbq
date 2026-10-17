"""Single source of truth for the fine-tune's class vocabulary.

The base model ships 15 document classes (<class_Text>, <class_Table>, ...). We append
our own; every tokenizer/model used for training or eval MUST add exactly this list, in
this order, or the token ids stop matching the checkpoint embeddings.
"""
CLASS_TOKENS = [
    "<class_Wall>",
    "<class_Door>",
    "<class_Window>",
    "<class_Room>",
]
# NOTE: <class_Dimension> (dim-sheet stage) is deliberately NOT here yet: ckpt-v4 was
# trained with these four ids, and adding a fifth now would shift the embedding rows and
# break --init. Add it only together with a from-scratch run or a fresh embedding.
CLASS_NAMES = [c[len("<class_"):-1] for c in CLASS_TOKENS]
