def main():
    # Lightweight smoke test to validate the model wiring and losses.
    from src.arc_model import (
        ArcPrizeModel,
        ArcTransformerModel,
        reconstruction_loss,
        reconstruction_loss_grid_ce,
        reconstruction_loss_grid_mse,
        direction_loss,
        predict_f,
    )
    import torch

    B, input_dim, latent_dim = 2, 16, 8
    model = ArcPrizeModel(input_dim, latent_dim)
    A = torch.randn(B, input_dim)
    Bx = torch.randn(B, input_dim)
    C = torch.randn(B, input_dim)
    Dx = torch.randn(B, input_dim)
    E = torch.randn(B, input_dim)

    z_a, a_rec = model(A)
    z_b, b_rec = model(Bx)
    z_c, c_rec = model(C)
    z_d, d_rec = model(Dx)

    loss_dir = direction_loss(z_a, z_b, z_c, z_d, same_task=True)
    loss_rec = (
        reconstruction_loss(A, a_rec)
        + reconstruction_loss(Bx, b_rec)
        + reconstruction_loss(C, c_rec)
        + reconstruction_loss(Dx, d_rec)
    ) / 4

    _, x_f_pred, _ = predict_f(model, A, Bx, E)
    print({
        "loss_dir": float(loss_dir.detach()),
        "loss_rec": float(loss_rec.detach()),
        "x_f_pred_shape": tuple(x_f_pred.shape),
    })

    # Transformer 2D: categorical grid example
    H, W, C = 8, 8, 10
    tmodel = ArcTransformerModel(H, W, d_model=64, nhead=8, num_layers_enc=2, num_layers_dec=2, num_classes=C)
    A = torch.randint(0, C, (B, H, W))
    Bx = torch.randint(0, C, (B, H, W))
    Cx = torch.randint(0, C, (B, H, W))
    Dx = torch.randint(0, C, (B, H, W))
    Ex = torch.randint(0, C, (B, H, W))
    z_a, a_rec_logits = tmodel(A)
    z_b, b_rec_logits = tmodel(Bx)
    z_c, c_rec_logits = tmodel(Cx)
    z_d, d_rec_logits = tmodel(Dx)
    loss_dir_t = direction_loss(z_a, z_b, z_c, z_d, same_task=True)
    loss_rec_t = (
        reconstruction_loss_grid_ce(A, a_rec_logits)
        + reconstruction_loss_grid_ce(Bx, b_rec_logits)
        + reconstruction_loss_grid_ce(Cx, c_rec_logits)
        + reconstruction_loss_grid_ce(Dx, d_rec_logits)
    ) / 4
    print({
        "t2d_loss_dir": float(loss_dir_t.detach()),
        "t2d_loss_rec": float(loss_rec_t.detach()),
    })


if __name__ == "__main__":
    main()
