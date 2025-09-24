import torch
from repo.orca_model import PatchEmbedding3D
import repo.dataloader as dataloader
import repo.model_utils as model_utils
# Default config in class:
# - patch_size=(2, 4, 4)
# - in_channels=63
# - expected_output_dim=128
# - time_steps=7, height=100, width=100

# batch_size = 2
# time_steps= 7

# x = torch.randn(batch_size, 7, 63, 100, 100)  # (N, C, T, H, W)

# embed = PatchEmbedding3D()  # use defaults
# y = embed(x)  # shape: (N, num_patches, expected_output_dim)

# print("Input shape:", x.shape)
# print("Output shape:", y.shape)  # e.g., (2, 36, 128) depending on padding/patching



import torch
from repo.orca_model import CNNEmbedder, CrossTuningModel, PredictionHead, PatchEmbedding3D, Prompt_Tuning_Model6_6

def main():
    # Settings consistent with the model code:
    # - Input image size: 100x100
    # - Kernel size: 10 → 10x10 patches → 100 patches total
    # - Hidden/Vision Transformer dim: 768
    batch_size = 2
    input_channels = 63
    image_size = 100
    kernel_size = 10
    seq_length = 2 
    vit_dim = 768
    # n_patches = (image_size // kernel_size) ** 2  # 100

    # Create args namespace with required attributes
    class Args:
        def __init__(self):
            self.prompt_dims = 256  # Prompt dimension size
            self.use_position_embedding = False

            self.historical_nwp_length = 2  # Number of historical NWP timesteps

            self.patch_size_t = 1  # Temporal patch size
            self.patch_size_h = 10  # Height patch size 
            self.patch_size_w = 10  # Width patch size
            self.use_position_embedding = True
            self.max_lead_time = 72
            self.start_lead_time = 0
            self.freeze = False
            self.image_size = 100
            self.data_dir = "/mnt/disk1/aiotlab/longnd/data/tc_data/basedyear_data_3days/data2"
            self.historical_data_length = 20

            
    args = Args()
    nwp_scaler, bt_scaler, n_fts = model_utils.fit_scalers_in_batches(args)
    train_dataset = dataloader.VITDataset6_6(data_dir= f"/mnt/disk1/aiotlab/longnd/data/tc_data/basedyear_data_3days/data2/train/data.npz",mode="train", args=args, nwp_scaler=nwp_scaler, bt_scaler= bt_scaler)
    breakpoint()
    cnn_embedder = orca_model.PatchEmbedding3D(in_channels=63, expected_output_dim=768 - 128, patch_size=10, n_timestep=args.historical_nwp_length, args=args)

    patch_size = (args.patch_size_t, args.patch_size_h, args.patch_size_w)
    n_fts = [input_channels]  # List containing number of input features
    # Build modules
    cnn_embedder = PatchEmbedding3D(in_channels=n_fts[0], expected_output_dim=768 - args.prompt_dims, patch_size=patch_size, n_timestep=args.historical_nwp_length, args=args)
    n_patches = cnn_embedder.n_patches
    prediction_head = PredictionHead(dim=vit_dim, n_patchs=n_patches + 2 )
    model = Prompt_Tuning_Model6_6(cnn_embed=cnn_embedder, body_model_name="vit", prediction_head=prediction_head, args=args)

    # Dummy input (batch, channels, H, W)
    nwp = torch.randn(batch_size, seq_length,input_channels, image_size, image_size)
    hres = torch.randn(batch_size, 10, 4)
    nwp_id = torch.randint(0, seq_length, (batch_size,))
    his = torch.randn(batch_size, 64)
    x = [nwp, his, nwp_id, hres]
    # Forward
    model.eval()
    with torch.no_grad():
        y = model(x)  # shape: (batch_size, 1)
    print("Output shape:", y.shape)
    print("Output:", y)

if __name__ == "__main__":
    main()