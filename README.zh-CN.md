# CrossSO

### Observe Less, Understand More: Cost-aware Cross-scale Observation for Remote Sensing Understanding

[[`arXiv`](https://arxiv.org/abs/2604.11415)]
[[`PDF`](https://arxiv.org/pdf/2604.11415)]
[[`English README`](README.md)]

CrossSO 通过低分辨率全局观测、稀疏高分辨率采样和跨区域表征预测，在控制观测成本的同时完成遥感理解。

<p align="center">
  <img src="assets/framework.png" width="100%" alt="CrossSO 框架">
</p>

## 最新动态

- **[2026/08/27]** CrossSO 权重已发布至 [Hugging Face](https://huggingface.co/xiehaoai/CrossSO)。
- **[2026/08/25]** 新增评测代码和示例 notebook。
- **[2026/08/15]** 发布中英文 README 和 GL-10M 数据准备工具。
- **[2026/04/13]** CrossSO 发布至 arXiv。

## 数据集

- **Sentinel-2/NAIP**：[Scale-Aware Recognition](https://github.com/ShreelekhaR/scale-aware)
- **EuroSAT**：[官方仓库](https://github.com/phelber/EuroSAT)；[Hugging Face 镜像](https://huggingface.co/datasets/torchgeo/eurosat)
- **BigEarthNet**：[官方存档](https://zenodo.org/records/12687186)；[RGB 数据集](https://huggingface.co/datasets/danielz01/BigEarthNet-S2-v1.0)
- **GL-10M**：即将发布至 Hugging Face

CrossSO 权重已发布至 [Hugging Face](https://huggingface.co/xiehaoai/CrossSO)。

## GL-10M 数据准备

```bash
pip install -e .
earthengine authenticate
python tools/prepare_gl10m.py --help
```

## 评测

```bash
pip install -e ".[eval]"
python tools/evaluate.py --help
```

评测路径见 `configs/`。本次更新包含 zero-shot retrieval、EuroSAT/BigEarthNet transfer 和 GL-10M evaluation。

## 示例

见 [`notebooks/crossso-method-demo.ipynb`](notebooks/crossso-method-demo.ipynb)。

## 许可证

<a href="https://creativecommons.org/licenses/by/4.0/"><img src="assets/cc-by.png" alt="CC BY 4.0" width="88"></a>

本项目采用[知识共享署名 4.0 国际许可协议](LICENSE)。第三方数据集与预训练模型遵循各自来源的许可协议。

## 致谢

感谢 [OpenStreetMap](https://www.openstreetmap.org/copyright) 贡献者提供地理数据与标注，感谢 [Google Earth Engine](https://earthengine.google.com/) 和 [Geofabrik](https://download.geofabrik.de/) 提供数据访问支持，并感谢 [GRAFT](https://graft.cs.cornell.edu/) 团队提供模型与相关资源。

## 引用

```bibtex
@article{xie2026observe,
  title={Observe Less, Understand More: Cost-aware Cross-scale Observation for Remote Sensing Understanding},
  author={Xie, Zhenghao and Xiao, Jing and Wang, Zhenqi and Ma, Kexin and Liao, Liang and Xia, Gui-song and Wang, Mi},
  journal={arXiv preprint arXiv:2604.11415},
  year={2026},
  doi={10.48550/arXiv.2604.11415}
}
```
