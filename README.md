# Generative Tensorial Reinforcement Learning (GENTRL) 
Supporting Information for the paper _"[Deep learning enables rapid identification of potent DDR1 kinase inhibitors](https://www.nature.com/articles/s41587-019-0224-x)"_.

The GENTRL model is a variational autoencoder with a rich prior distribution of the latent space. We used tensor decompositions to encode the relations between molecular structures and their properties and to learn on data with missing values. We train the model in two steps. First, we learn a mapping of a chemical space on the latent manifold by maximizing the evidence lower bound. We then freeze all the parameters except for the learnable prior and explore the chemical space to find molecules with a high reward.

![GENTRL](images/gentrl.png)


## Repository
In this repository, we provide an implementation of a GENTRL model with an example trained on a [MOSES](https://github.com/molecularsets/moses) dataset.

To run the training procedure,
1. [Install RDKit](https://www.rdkit.org/docs/Install.html) to process molecules
2. Install GENTRL model: `python setup.py install`
3. Install MOSES from the [repository](https://github.com/molecularsets/moses)
4. Run the [pretrain.ipynb](./examples/pretrain.ipynb) to train an autoencoder
5. Run the [train_rl.ipynb](./examples/train_rl.ipynb) to optimize a reward function
