import torch
import torch.nn as nn

from layers import SegmentationUnet

# Function to register hooks and calculate activation sizes
def get_activation_size(model, input_tensor):
    total_size = 0  # Initialize total size of activations

    # Hook function to calculate size of activations
    def hook_fn(module, input, output):
        nonlocal total_size
        # Calculate the number of elements in the output (activation)
        activation_size = output.numel()  # numel() returns total number of elements
        total_size += activation_size
        print(type(module), activation_size)

    # Register hooks on each module
    hooks = []
    for layer in model.modules():
        if not isinstance(layer, nn.Sequential) and not isinstance(layer, nn.ModuleList) and layer != model:
            hooks.append(layer.register_forward_hook(hook_fn))

    # Perform a forward pass to trigger the hooks
    t = torch.zeros((1,))
    model(t, input_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return total_size


model = SegmentationUnet(num_classes=4, dim=32, num_steps=1000)
input_tensor = torch.randint(0, 4, size=(1, 1, 512, 512))  # Batch size of 1, and input size (3, 8, 8)

total_activation_size = get_activation_size(model, input_tensor)
print(f"Total size of intermediate activations: {total_activation_size} elements")
""" from layers import SegmentationUnet
model = SegmentationUnet(num_classes=4, dim=32, num_steps=1000)
print(f"Total number of parameters: {sum(p.numel() for p in model.parameters())}") """

