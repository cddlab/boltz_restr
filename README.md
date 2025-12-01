# Boltz with Restraint-Guided Inference

This repository provides an extended version of Boltz-1/2 with **restraint-guided inference** .

Current *Restraint-guided inference* enables ligand conformer restraints and distance restraints.


<table>
  <tr>
    <td><img src="docs/conformer.gif" alt="conformer" width="300"></td>
    <td><img src="docs/distance_qbp.gif" alt="distance" width="300"></td>
    <td><img src="docs/distance_mdm2_p53.gif" alt="distance2" width="300"></td>
  </tr>
</table>


## 🚀 Quick Start (No Installation Required)

Try the method directly in Google Colab without any installation:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]()

## 📋 Key Features

- **No model retraining required** - works with existing Boltz-1 weights
- **GPU acceleration** for restraint calculations
- **Distance restraints**
  - **Sampling domain motion of Protein**
  - **Sampling dissociation pathway of Ligand**
  - **Improving Protein-Ligand docking pose**
  - **Guiding Ligand to other binding sites**
- **Conformer restraints**
  - **100% chirality reproduction** for input molecular structures
  - **Significant improvement** in bond lengths and angle geometries
  - **Maintains protein structure quality** while fixing ligand stereochemistry

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (recommended for performance)

```bash
git clone https://github.com/cddlab/boltz_restr.git
cd boltz_restr
uv venv
uv pip install -e ".[cuda]"
uv pip install torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cu128.html # if using CUDA 12.8
```

## ⚙️ Configuration

### Basic Usage

To enable restraint-guided inference, modify your configuration YAML file:

#### 1. Enable conformer Restraints for Ligands

Add `conformer_restraints: true` at the same level as your ligand CCD code or SMILES:

```yaml
sequences:
  - protein:
      id: A
      sequence: "MKFLVL..."
  - ligand:
      ccd: "ATP"  # or smiles: "CC(C)CC..."
      conformer_restraints: true  # Add this line
```

#### 2. Configure Restraint Parameters

Add a top-level `restraints_config` section:

```yaml
restraints_config:
  verbose: true
  max_iter: 1000
  method: "CG"
  start_sigma: 999999
  gpu: true

  distance_restraints_config:
    - atom_selection1: "chain A1"
      atom_selection2: "chain B1"
      harmonic:
        target_distance: 50

  conformer_restraints_config:
    bond:
      weight: 1
    angle:
      weight: 1
    chiral:
      weight: 1
    vdw:
      weight: 1
```

### Complete Configuration Example

```yaml
restraints_config:
  verbose: true
  max_iter: 1000
  method: "CG"
  start_sigma: 999999
  gpu: true

  distance_restraints_config:
    - atom_selection1: "chain A1"
      atom_selection2: "chain B1"
      # NOTE:
      # RESERVED_KEYWORDS = {"and", "or", "not", "to", "resid", "index", "chain", "(", ")"}
      # resid and index start from 0
      # Group of atom_selection1 is fixed
      # Group of atom_selection2 is moved
      # boltz_restr calculates center of mass distance between two groups

      # harmonic: Adds a quadratic penalty to enforce the distance to be equal to target_distance.
      harmonic:
        target_distance: 50

      # flat-bottomed: Adds a penalty only when the distance is outside the range [target_distance1, target_distance2].
      # flat-bottomed:
      #   target_distance1: 26
      #   target_distance2: 30

      # flat-bottomed1: Adds a penalty when the distance is smaller than target_distance1.
      # flat-bottomed1:
      #   target_distance1: 30

      # flat-bottomed2: Adds a penalty when the distance is larger than target_distance2.
      # flat-bottomed2:
      #   target_distance2: 20

    - atom_selection1: "(resid 1 to 109) or (resid 264 to 309)"
      atom_selection2: "(resid 144 to 258) or (resid 316 to 370)"

      harmonic:
        target_distance: 30

  conformer_restraints_config:
    bond:
      weight: 1
    angle:
      weight: 1
    chiral:
      weight: 1
    vdw:
      weight: 1
```

### Configuration Options

#### Parameters

- **`weight`**: Relative weight for each restraint type (default: 1)
- **`start_sigma`**: Sigma threshold below which restraints are applied (default: 1.0)
  - Highly recommended to use large values(e.g. 999999) if you use distance restraints
- **`gpu`**: Enable GPU-accelerated constraint calculations (default: false)
  - Highly recommended for large ligands or multiple diffusion samples

#### Restraint Combinations

You can use different combinations of restraints:
- All restraints
```yaml
restraints_config:
  verbose: true
  max_iter: 1000
  method: "CG"
  start_sigma: 999999
  gpu: true

  distance_restraints_config:
    - atom_selection1: "chain A1"
      atom_selection2: "chain B1"
      harmonic:
        target_distance: 50
    - atom_selection1: "(resid 1 to 109) or (resid 264 to 309)"
      atom_selection2: "(resid 144 to 258) or (resid 316 to 370)"
      harmonic:
        target_distance: 30

  conformer_restraints_config:
    bond:
      weight: 1
    angle:
      weight: 1
    chiral:
      weight: 1
    vdw:
      weight: 1
```

- Only distance restraints
```yaml
restraints_config:
  verbose: true
  max_iter: 1000
  method: "CG"
  start_sigma: 999999
  gpu: true

  distance_restraints_config:
    - atom_selection1: "chain A1"
      atom_selection2: "chain B1"
      harmonic:
        target_distance: 50
    - atom_selection1: "(resid 1 to 109) or (resid 264 to 309)"
      atom_selection2: "(resid 144 to 258) or (resid 316 to 370)"
      harmonic:
        target_distance: 30
```

- Only conformer restraints
```yaml
restraints_config:
  verbose: true
  max_iter: 1000
  method: "CG"
  start_sigma: 999999
  gpu: true

  conformer_restraints_config:
    bond:
      weight: 1
    angle:
      weight: 1
    chiral:
      weight: 1
    vdw:
      weight: 1
```


## 🚧 TODO
- [x] Enable conformer-restraints and distance-restraints
- [x] Enable GPU acceleration
- [x] Enable multiple distance-restraints
- [ ] Code refactoring


## 📚 Citation

If you use this work in your research, please cite:

- For conformer-restraints and distance-restraints
```bibtex
```

- For original restraints-guided inference(conformer-restraints)
```bibtex
@article{ishitani2025improving,
  title={Improving Stereochemical Limitations in Protein--Ligand Complex Structure Prediction},
  author={Ishitani, Ryuichiro and Moriwaki, Yoshitaka},
  journal={ACS Omega},
  year={2025},
  publisher={ACS Publications}
}

@article{ishitani2025improving,
  title={Improving Stereochemical Limitations in Protein-Ligand Complex Structure Prediction},
  author={Ishitani, Ryuichiro and Moriwaki, Yoshitaka},
  journal={bioRxiv},
  year={2025},
  doi={10.1101/2025.03.25.645362v2}
}
```

---

<details> <summary> Original Boltz README.md </summary>

<div align="center">
  <div>&nbsp;</div>
  <img src="docs/boltz2_title.png" width="300"/>
  <img src="https://model-gateway.boltz.bio/a.png?x-pxid=bce1627f-f326-4bff-8a97-45c6c3bc929d" />

[Boltz-1](https://doi.org/10.1101/2024.11.19.624167) | [Boltz-2](https://doi.org/10.1101/2025.06.14.659707) |
[Slack](https://join.slack.com/t/boltz-community/shared_invite/zt-37uc4m8t2-gbbph6ka704ORcDCHLlFKg) <br> <br>
</div>



![](docs/boltz1_pred_figure.png)


## Introduction

Boltz is a family of models for biomolecular interaction prediction. Boltz-1 was the first fully open source model to approach AlphaFold3 accuracy. Our latest work Boltz-2 is a new biomolecular foundation model that goes beyond AlphaFold3 and Boltz-1 by jointly modeling complex structures and binding affinities, a critical component towards accurate molecular design. Boltz-2 is the first deep learning model to approach the accuracy of physics-based free-energy perturbation (FEP) methods, while running 1000x faster — making accurate in silico screening practical for early-stage drug discovery.

All the code and weights are provided under MIT license, making them freely available for both academic and commercial uses. For more information about the model, see the [Boltz-1](https://doi.org/10.1101/2024.11.19.624167) and [Boltz-2](https://doi.org/10.1101/2025.06.14.659707) technical reports. To discuss updates, tools and applications join our [Slack channel](https://join.slack.com/t/boltz-community/shared_invite/zt-37uc4m8t2-gbbph6ka704ORcDCHLlFKg).

## Installation

> Note: we recommend installing boltz in a fresh python environment

Install boltz with PyPI (recommended):

```
pip install boltz[cuda] -U
```

or directly from GitHub for daily updates:

```
git clone https://github.com/jwohlwend/boltz.git
cd boltz; pip install -e .[cuda]
```

If you are installing on CPU-only or non-CUDA GPus hardware, remove `[cuda]` from the above commands. Note that the CPU version is significantly slower than the GPU version.

## Inference

You can run inference using Boltz with:

```
boltz predict input_path --use_msa_server
```

`input_path` should point to a YAML file, or a directory of YAML files for batched processing, describing the biomolecules you want to model and the properties you want to predict (e.g. affinity). To see all available options: `boltz predict --help` and for more information on these input formats, see our [prediction instructions](docs/prediction.md). By default, the `boltz` command will run the latest version of the model.


### Binding Affinity Prediction
There are two main predictions in the affinity output: `affinity_pred_value` and `affinity_probability_binary`. They are trained on largely different datasets, with different supervisions, and should be used in different contexts. The `affinity_probability_binary` field should be used to detect binders from decoys, for example in a hit-discovery stage. It's value ranges from 0 to 1 and represents the predicted probability that the ligand is a binder. The `affinity_pred_value` aims to measure the specific affinity of different binders and how this changes with small modifications of the molecule. This should be used in ligand optimization stages such as hit-to-lead and lead-optimization. It reports a binding affinity value as `log(IC50)`, derived from an `IC50` measured in `μM`. More details on how to run affinity predictions and parse the output can be found in our [prediction instructions](docs/prediction.md).

## Authentication to MSA Server

When using the `--use_msa_server` option with a server that requires authentication, you can provide credentials in one of two ways. More information is available in our [prediction instructions](docs/prediction.md).

## Evaluation

⚠️ **Coming soon: updated evaluation code for Boltz-2!**

To encourage reproducibility and facilitate comparison with other models, on top of the existing Boltz-1 evaluation pipeline, we will soon provide the evaluation scripts and structural predictions for Boltz-2, Boltz-1, Chai-1 and AlphaFold3 on our test benchmark dataset, and our affinity predictions on the FEP+ benchmark, CASP16 and our MF-PCBA test set.

![Affinity test sets evaluations](docs/pearson_plot.png)
![Test set evaluations](docs/plot_test_boltz2.png)


## Training

⚠️ **Coming soon: updated training code for Boltz-2!**

If you're interested in retraining the model, currently for Boltz-1 but soon for Boltz-2, see our [training instructions](docs/training.md).


## Contributing

We welcome external contributions and are eager to engage with the community. Connect with us on our [Slack channel](https://join.slack.com/t/boltz-community/shared_invite/zt-37uc4m8t2-gbbph6ka704ORcDCHLlFKg) to discuss advancements, share insights, and foster collaboration around Boltz-2.

On recent NVIDIA GPUs, Boltz leverages the acceleration provided by [NVIDIA  cuEquivariance](https://developer.nvidia.com/cuequivariance) kernels. Boltz also runs on Tenstorrent hardware thanks to a [fork](https://github.com/moritztng/tt-boltz) by Moritz Thüning.

## License

Our model and code are released under MIT License, and can be freely used for both academic and commercial purposes.


## Cite

If you use this code or the models in your research, please cite the following papers:

```bibtex
@article{passaro2025boltz2,
  author = {Passaro, Saro and Corso, Gabriele and Wohlwend, Jeremy and Reveiz, Mateo and Thaler, Stephan and Somnath, Vignesh Ram and Getz, Noah and Portnoi, Tally and Roy, Julien and Stark, Hannes and Kwabi-Addo, David and Beaini, Dominique and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction},
  year = {2025},
  doi = {10.1101/2025.06.14.659707},
  journal = {bioRxiv}
}

@article{wohlwend2024boltz1,
  author = {Wohlwend, Jeremy and Corso, Gabriele and Passaro, Saro and Getz, Noah and Reveiz, Mateo and Leidal, Ken and Swiderski, Wojtek and Atkinson, Liam and Portnoi, Tally and Chinn, Itamar and Silterra, Jacob and Jaakkola, Tommi and Barzilay, Regina},
  title = {Boltz-1: Democratizing Biomolecular Interaction Modeling},
  year = {2024},
  doi = {10.1101/2024.11.19.624167},
  journal = {bioRxiv}
}
```

In addition if you use the automatic MSA generation, please cite:

```bibtex
@article{mirdita2022colabfold,
  title={ColabFold: making protein folding accessible to all},
  author={Mirdita, Milot and Sch{\"u}tze, Konstantin and Moriwaki, Yoshitaka and Heo, Lim and Ovchinnikov, Sergey and Steinegger, Martin},
  journal={Nature methods},
  year={2022},
}
```

</details>
