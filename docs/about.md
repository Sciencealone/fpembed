# About

Use this page to find the research behind FPembed, the software and methods it builds on, and the citation for the archived release. It also records the license and the support routes.

## Original Concept (eMFP)

The original concept of compressing molecular fingerprints with weighted binary
masking comes from this work on Morgan fingerprints:

> Nuñez-Andrade, E. A., Vidal-Daza, I., Gomez-Bombarelli, R., Ryan, J. W., and
> Martin-Martinez, F. J. (2025). Embedded Morgan Fingerprints for more efficient
> molecular property predictions with machine learning. ChemRxiv preprint.
> https://doi.org/10.26434/chemrxiv-2025-6hfp8

Original concept repository: https://github.com/MMLabCodes/eMFP

```bibtex
@article{nunez2025emfp,
  author  = {Nu{\~n}ez-Andrade, Emilio A. and Vidal-Daza, Isaac and Gomez-Bombarelli, Rafael and Ryan, James W. and Martin-Martinez, Francisco J.},
  title   = {Embedded {Morgan} Fingerprints for more efficient molecular property predictions with machine learning},
  journal = {ChemRxiv},
  year    = {2025},
  doi     = {10.26434/chemrxiv-2025-6hfp8},
  note    = {Preprint}
}
```

### Software and methods

- RDKit: https://www.rdkit.org/
- scikit-fingerprints: https://github.com/scikit-fingerprints/scikit-fingerprints
- SELFIES: https://github.com/aspuru-guzik-group/selfies
- Random projection and the Johnson-Lindenstrauss lemma: W. B. Johnson and J. Lindenstrauss, "Extensions of Lipschitz mappings into a Hilbert space", Contemporary Mathematics 26, 1984.
- Fast Walsh-Hadamard transform: a standard divide-and-conquer transform over Sylvester-ordered Hadamard matrices.

### Citing FPembed

The archived software record has DOI [10.5281/zenodo.21447996](https://doi.org/10.5281/zenodo.21447996).

## License and Support

FPembed is released under the MIT license. The authoritative license text is in [LICENSE](https://github.com/Sciencealone/fpembed/blob/main/LICENSE).

Legal, trademark, commercial, and support details are maintained in the README: [License](https://github.com/Sciencealone/fpembed/blob/main/README.md#license), [Legal & Trademark Notice](https://github.com/Sciencealone/fpembed/blob/main/README.md#legal--trademark-notice), [Commercial & Research Services](https://github.com/Sciencealone/fpembed/blob/main/README.md#commercial--research-services), [AI Disclosure](https://github.com/Sciencealone/fpembed/blob/main/README.md#ai-disclosure), and [Support](https://github.com/Sciencealone/fpembed/blob/main/README.md#support).

The project is provided as-is and may be updated over time. Questions and bug reports belong in the project issue tracker.
