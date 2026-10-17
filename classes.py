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
    "<class_Dimension>",   # dimension strings / witness-line spans (ft-in reading)
]
CLASS_NAMES = [c[len("<class_"):-1] for c in CLASS_TOKENS]
