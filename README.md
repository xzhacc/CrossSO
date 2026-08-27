# CrossSO

### Observe Less, Understand More: Cost-aware Cross-scale Observation for Remote Sensing Understanding

[[`arXiv`](https://arxiv.org/abs/2604.11415)]
[[`PDF`](https://arxiv.org/pdf/2604.11415)]
[[`中文 README`](README.zh-CN.md)]

CrossSO combines global low-resolution observations, sparse high-resolution sampling, and cross-region representation prediction for cost-aware remote sensing understanding.

<p align="center">
  <img src="assets/framework.png" width="100%" alt="CrossSO framework">
</p>

## News

- **[2026/08/27]** CrossSO weights are released on [Hugging Face](https://huggingface.co/xiehaoai/CrossSO).
- **[2026/08/25]** Evaluation code and the demo notebook are added.
- **[2026/08/15]** The bilingual README and GL-10M data-preparation tools are released.
- **[2026/04/13]** CrossSO is available on arXiv.

## Datasets

- **Sentinel-2/NAIP**: [Scale-Aware Recognition](https://github.com/ShreelekhaR/scale-aware)
- **EuroSAT**: [official repository](https://github.com/phelber/EuroSAT); [Hugging Face mirror](https://huggingface.co/datasets/torchgeo/eurosat)
- **BigEarthNet**: [official archive](https://zenodo.org/records/12687186); [RGB dataset](https://huggingface.co/datasets/danielz01/BigEarthNet-S2-v1.0)
- **GL-10M**: coming soon on Hugging Face

CrossSO weights are available on [Hugging Face](https://huggingface.co/xiehaoai/CrossSO).

## GL-10M Data Preparation

```bash
pip install -e .
earthengine authenticate
python tools/prepare_gl10m.py --help
```

## Evaluation

```bash
pip install -e ".[eval]"
python tools/evaluate.py --help
```

Evaluation paths are defined in `configs/`. This update includes zero-shot retrieval, EuroSAT/BigEarthNet transfer, and GL-10M evaluation.

## Demo

See [`notebooks/crossso-method-demo.ipynb`](notebooks/crossso-method-demo.ipynb).

## License

<a href="https://creativecommons.org/licenses/by/4.0/"><img src="assets/cc-by.png" alt="CC BY 4.0" width="88"></a>

This project is licensed under the [Creative Commons Attribution 4.0 International License](LICENSE). Third-party datasets and pretrained models remain subject to their respective terms.

## Acknowledgements

We thank the [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors for geographic data and annotations, [Google Earth Engine](https://earthengine.google.com/) and [Geofabrik](https://download.geofabrik.de/) for data access, and the [GRAFT](https://graft.cs.cornell.edu/) team for its models and resources.

## Citation

```bibtex
@article{xie2026observe,
  title={Observe Less, Understand More: Cost-aware Cross-scale Observation for Remote Sensing Understanding},
  author={Xie, Zhenghao and Xiao, Jing and Wang, Zhenqi and Ma, Kexin and Liao, Liang and Xia, Gui-song and Wang, Mi},
  journal={arXiv preprint arXiv:2604.11415},
  year={2026},
  doi={10.48550/arXiv.2604.11415}
}
```
