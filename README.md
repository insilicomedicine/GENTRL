# Generative Tensorial Reinforcement Learning (GENTRL) 
Supporting Information for the paper _"[Deep learning enables rapid identification of potent DDR1 kinase inhibitors](https://www.nature.com/articles/s41587-019-0224-x)"_.

The GENTRL model is a variational autoencoder with a rich prior distribution of the latent space. We used tensor decompositions to encode the relations between molecular structures and their properties and to learn on data with missing values. We train the model in two steps. First, we learn a mapping of a chemical space on the latent manifold by maximizing the evidence lower bound. We then freeze all the parameters except for the learnable prior and explore the chemical space to find molecules with a high reward.

![GENTRL](images/gentrl.png)

# Installation

## Step 1 :
Make a new conda environment and install RDKit.
```
conda create -c rdkit -n my-rdkit-env rdkit
```
Then activate this new environment.
```
conda activate my-rdkit-env
```
*Note :*  Make sure that the python3 version is 3.5 or higher and pip3 is installed

## Step 2 :
Inside this environment install GENTRL.
```
cd <Path_to_GENTRL_folder>
python3 setup.py install
```

## Step 3 : (Optional)
Making a new **Kernel** for jupyter notebook is recommended. For making a new kernel please follow these steps.
```
python3 -m pip install ipykernel
python3 -m ipykernel install --user --name rdkit_kernel
```
Now when you open jupyter notebook. Go to **Change Kernel** > **rdkit_kernel**

With these the installation is over and now we are ready to run the examples provided in the Repo.

# Explanation

## pretrain.ipynb
This notebook trains a Variational Auto Encoder (VAE) with TTLP prior distribution to encode SMILES strings onto the latent space. In this step, VAE maps the structural properties of each training molecule to a latent code.


## train_rl.ipynb
This notebook applies reinforcement learning to optimize a reward function. In this step, encoder and decoder are fixed, and the model learns only the learnable prior.

## sampling.ipynb
This notebook generates new molecules in the form of SMILES strings. Below we show examples of generated molecules (more samples [here](https://github.com/Bibyutatsu/GENTRL/blob/master/images/Sampling_big.png)).

![Sampling](https://github.com/Bibyutatsu/GENTRL/blob/master/images/Sampling.jpeg)
