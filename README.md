# SimpleVAE implementation in Pytorch - with added interactive visualization
![interactive](interactive_plot.png)

Code is complete with small explanations of the VAE.
This latent space is a bit different from the plain autoencoder. The sampling around encoded points causes the latent space to be more continuous. Additionally, the objective to put the means as close to 0 and deviations as close to 1 as possible (KL loss) makes the latent space more uniform.


## How to run
1. Run `python3 run.py` to train the model (this creates `model.pt` in the `checkpoints` folder.
2. Run `python3 visualize.py` to see the latent space for the model you just trained.

There are also pretrained weights available, so you can just do step 2 if you want.
