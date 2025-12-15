import yaml
import pandas as pd 

with open("configs/exp.yaml", "r") as f:
    cfg = yaml.safe_load(f)
# ----------------------- Main script -----------------------

# Paths: replace with your local paths (Kaggle / mount)
mat_path = "/kaggle/input/hyperspectral-image-sensing-dataset-ground-truth/Indian_pines_corrected.mat"
gt_path = "/kaggle/input/hyperspectral-image-sensing-dataset-ground-truth/Indian_pines_gt.mat"

cube, X, Y = load_hsi(mat_path, gt_path)
H, W, B = cube.shape
print(f"Loaded cube H={H}, W={W}, B={B}, labeled pixels: {X.shape[0] if X is not None else 0}")

# hyperparams (tweak to taste)
# window_size =5
# latent_dim = 256
# decoder_dropout = 0.0
# decoder_hidden_mult = 8
# C_init_scale = 1e-1        # larger initial C helps avoid tiny values
# C_lr_mult = 50.0           # give C a larger learning rate
# alpha = 10.0                # overwritten by auto-scale if enabled
# beta = 1e-4                # weaker Frobenius reg to avoid shrinking C too hard
# lambda_l1 = 5e-4
# lambda_tv = 5e-4
# lr = 1e-3
# epochs = 20
# batch_size =16

model = HybridModel(window_size=window_size, H=H, W=W, latent_dim=latent_dim, use_attention=False,
                    decoder_dropout=decoder_dropout, decoder_hidden_mult=decoder_hidden_mult)

print("Training Hybrid 3D-CNN + Self-Expression model (minibatch-safe, memory_cautious=True)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Use the train_model function defined above (it returns model, Z, C, A)
model, Z, C_final, A_raw, history, recon_cube = train_model(
    model,
    cube,
    epochs=epochs,
    batch_size=batch_size,
    lr=lr,
    alpha=alpha, beta=beta, device=device,
    verbose=True,
    C_lr_mult=C_lr_mult,
    C_init_scale=1e-1,
    lambda_l1=lambda_l1,
    lambda_tv=lambda_tv,
    val_frac=0.15,
    memory_cautious=True,
    auto_scale_alpha=True,
    desired_self_ratio=1,
    normalize_Z_for_C=False,

    # new C-control args:
    lambda_C_l1 = 1e-3,     # weight for L1 on off-diagonal C (additive in loss)
    use_proximal_C =True,  # apply soft-thresholding on C.data after each step
    prox_tau_scale = 0.2
)

# Output & diagnostics
print("\nA_raw (first 8x8 block):\n", np.round(A_raw[:8,:8], 8))
A_display = A_raw / (A_raw.max() + 1e-12)
print("\nA_display (first 8x8 block):\n", np.round(A_display[:8,:8], 6))

band_imp = get_band_importance_from_dict(A_raw, method='l2')
plot_importance(band_imp, title="Band importance (from A, L2 normalized)")
# qqqqq
# plot epoch histories
plot_epoch_history(history, savefile='training_history.png')

# compute per-band RMSE and SNR (use all pixels mask)
rmse_per_band, snr_db = compute_per_band_rmse_and_snr(cube, recon_cube, mask=None)
plot_rmse_snr(rmse_per_band, snr_db, importance=band_imp, savefile='rmse_snr.png')

k = 20
topk_simple = compute_band_scores_all(A_raw, k=k)
for k in topk_simple:
    print(f"Top-k  for {k}(simple affinity-sum):", sorted(topk_simple[k].tolist()))
    test_selected_bands(X, Y, topk_simple[k])

    # if SKLEARN_AVAILABLE:
    #     n_clusters = k
    #     topk_clustered, band_labels = select_topk_bands_spectral(A_raw, n_clusters=n_clusters)
    #     print("Top-k (clustered representatives):", sorted(topk_clustered))
    #     test_selected_bands(X, Y, topk_clustered)
    #     plot_clusters(band_labels, B)
    # else:
    #     print("sklearn not available — skipping clustered selection.")

    # Save CSV of band importance & selected bands
    # df = pd.DataFrame({'band': np.arange(B), 'importance': band_imp, 'rmse': rmse_per_band, 'snr_db': snr_db})
    # df_sorted = df.sort_values('importance', ascending=False).reset_index(drop=True)
    # csv_out = "band_importance_and_selection_with_metrics.csv"
    # df_sorted.to_csv(csv_out, index=False)
    # np.savetxt("topk_simple.txt", topk_simple, fmt="%d")
    # if SKLEARN_AVAILABLE:
    #     np.savetxt("topk_clustered.txt", topk_clustered, fmt="%d")
    # print(f"Saved band importance to {csv_out} and selected indices to topk_simple.txt (and topk_clustered.txt if available).")

    # # show high-noise bands images
    # show_high_noise_band_images(cube, recon_cube, rmse_per_band, top_n=6, save_dir=None)

# small visualization loop for sample bands (optional)
for idx in range (4, min(B,200), 20):
    try:
            ds_vis = BandPatchDataset(cube, window_size=window_size)
            patch_vis, target_vis, bidx = ds_vis[idx]
            # compute recon for that band using the encoder+decoder pipeline
            model.eval()
            with torch.no_grad():
                # prepare single patch batch
                patch_batch = patch_vis.unsqueeze(0).to(next(model.parameters()).device)
                Z_single = model.encoder(patch_batch)
                recon_single = model.decoder(Z_single).cpu().numpy().reshape(H, W)
                targ_single = target_vis.cpu().numpy().reshape(H, W)
            plt.figure(figsize=(8,4))
            plt.subplot(1,2,1); plt.title(f"target band {idx}"); plt.imshow(targ_single, cmap='viridis'); plt.colorbar()
            plt.subplot(1,2,2); plt.title(f"recon band {idx}"); plt.imshow(recon_single, cmap='viridis'); plt.colorbar()
            plt.suptitle(f"Band {idx} target vs recon (visual check)")
            plt.show()
    except Exception:
            pass