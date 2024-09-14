# Dynamical Causality: A Dynamical View of the Question of Why

This repository hosts the code release for the paper ["A Dynamical View of the Question of Why"](https://arxiv.org/abs/2402.10240), published at [**ICLR 2024**](https://iclr.cc/Conferences/2024).

This work was done by 

* [**Mehdi Fatemi**](https://www.linkedin.com/in/fatemi/) ([Wand.ai](https://wand.ai/)) 

* [**Sindhu Gowda**](https://sindhucmgowda.github.io/) (University of Toronto and Vector Institute).

We release a flexible codebase, which enables replicating the experimental results (and plots) presented in the paper. The codebase can be used for further study/analysis of various forms. As we believe this paper provides a new avenue for further research, this codebase can also be used as the basis for pursuing various new ideas and possible future work. We do encourage collaborations. 

## [LICENSE](https://github.com/fatemi/dynamical-causality/blob/main/LICENSE)

## Citing

If you make use of our work, please use the citation information below:

```
@inproceedings{fatemi2024dynamicalcausality,
      title={A Dynamical View of the Question of Why}, 
      author={Mehdi Fatemi and Sindhu Gowda},
      booktitle={The Twelfth International Conference on Learning Representations (ICLR)},
      year={2024},
      url={https://arxiv.org/abs/2402.10240}, 
}
```

## Getting started

Install the following python packages:

```
pip install click
pytorch >= 2.3 (https://pytorch.org/)
OpenCV (CV2 == 4.9.0)
gym == 0.26.2 (old versions should work)
```

## Atari Game of Pong

In order to replicate the Pong results in the paper (Section 5 --> **Atari Game of Pong**), you will need to first train the Q-Network. To make life easier, we provide a fully trained network, with which you can simply jump to the analysis. The trained network is placed in the following folder:```./atari/results/pong0```This is the default folder which is also made if you run the training code from scratch. In the folder above, you will find both the config file from training, as well as the trained model at `./atari/results/pong0/ai/q_network_weights_201.pt`

To start the analysis, open the notebook `./atari/analysis.ipynb` and run its cells. It will make a new folder `figs` inside the above folder (`./atari/results/pong0/figs`) and dump all the figures inside it. Note the file `full_46_56.pdf` which is what has appeared in the paper. Additionally, the last cell also makes frames of a video for better visuals. You may use the file `video.py` to make a gif from these frames (or other tools of your choice).

In order to train the Q-Network for Pong, you will need to run the following command from the `./atari` folder:
```
python train.py
```
You can pass various training parameters to the above command using the flag `-o` (e.g., `-o num_epochs 100`), or rather adjust them directly inside the `./atari/config_atari.yaml` file. The trainer will make a folder akin to the one provided above and dump the trained model at the end of each epoch (1 million frames by default). The one provided above is after 200 epochs. 

### NOTE: 

We have deliberately provided the Q-Network training in a very basic and clean form without using more advanced algorithms, over-engineering, or additional bells and whistles for speeding up, etc. This offers clarity and readibility, as well as ease of scrutiny. If you would like to use a Q-Network trained by other [Pytorch] frameworks, **make sure to use $\gamma = 1$** and not 0.99, as it is the common/default practice. Next, you will need to implement something similar to the `get_grad` [method in the RL class](https://github.com/fatemi/dynamical-causality/blob/main/atari/rl.py#L139). Finally, you should modify `analysis.ipynb` accordingly [it is a bit of extra work but should not be difficult]. 

## Real-world Diabetes Simulator

Will be provided soon! Stay tuned...
