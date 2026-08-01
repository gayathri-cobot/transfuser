import torch
import gc

# Delete tensors/models you no longer need first
# del model, optimizer, inputs  # whatever variables are holding GPU memory
gc.collect()
torch.cuda.empty_cache()

print(torch.cuda.memory_summary())
# or more concise:
print(f"Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
