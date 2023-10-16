Code accompanying the EMNLP 2023 publication "Recurrent Neural Language Models as Probabilistic Finite-state Automata".

## Getting started with the code

Clone the repository:

```bash
$ git clone git@github.com:rycolab/weighted-minsky.git
$ cd weighted-minsky
```

At this point it may be beneficial to create a new [Python virtual environment](https://docs.python.org/3.8/tutorial/venv.html). There are multiple solutions for this step, including [Miniconda](https://docs.conda.io/en/latest/miniconda.html). We aim at Python 3.10 version and above.

Then you install the package _in editable mode_:

```bash
$ pip install -e .
```

We use [black](https://github.com/psf/black) and [flake8](https://flake8.pycqa.org/en/latest/) to lint the code, [pytype](https://github.com/google/pytype) to check whether the types agree, and [pytest](https://docs.pytest.org) to unit test the code.


---

# Citation
```bibtex
@article{svete2023recurrent,
      title={Recurrent Neural Language Models as Probabilistic Finite-state Automata}, 
      author={Anej Svete and Ryan Cotterell},
      year={2023},
      eprint={2310.05161},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      journal = {arXiv preprint arXiv:2310.05161},
      url={https://arxiv.org/abs/2310.05161}
}
```