---
layout: paper
paper: ID-Reveal Identity-aware DeepFake Video Detection
github_url: https://github.com/grip-unina/id-reveal
authors:  
  - name: Fabrizio Guillaro
    link: https://www.grip.unina.it/members/guillaro
    index: 1
  - name: Davide Cozzolino
    link: https://www.grip.unina.it/members/cozzolino
    index: 2
  - name: Avneesh Sud
    link: 
    index: 3
  - name: Nicholas Dufour
    link: 
    index: 4
  - name: Luisa Verdoliva
    link: https://www.grip.unina.it/members/verdoliva
    index: 5
affiliations: 
  - name: University Federico II of Naples, Italy
    index: 1
  - name: Google AI
    index: 2
links:
    arxiv: 
    code: https://github.com/grip-unina/TruFor
---

<!-- <center><img src="./header.jpg" alt="header" height="200" /></center> -->
In this paper we present TruFor, a forensic framework that can be applied to a large variety of image manipulation methods, from classic cheapfakes to more recent manipulations based on deep learning. We rely on the extraction of both high-level and low-level traces through a transformer-based fusion architecture that combines the RGB image and a learned noise-sensitive fingerprint. The latter learns to embed the artifacts related to the camera internal and external processing by training only on real data in a self-supervised manner. Forgeries are detected as deviations from the expected regular pattern that characterizes each pristine image. Looking for anomalies makes the approach able to robustly detect a variety of local manipulations, ensuring generalization. In addition to a pixel-level localization map and a whole-image integrity score, our approach outputs a reliability map that highlights areas where localization predictions may be error-prone. This is particularly important in forensic applications in order to reduce false alarms and allow for a large scale analysis. Extensive experiments on several datasets show that our method is able to reliably detect and localize both cheapfakes and deepfakes manipulations outperforming state-of-the-art works. 

## News

<!-- *   2022-12-21: Paper is uploaded on arXiv -->

## Links

https://github.com/grip-unina/TruFor

## Requirements

## Bibtex
<!-- 
```javascript
@article{Guillaro2022_trufor,
  title={},
  author={},
  journal={arXiv preprint arXiv:},
  year={2022}
}
```
 -->
 
 ## License
 
 ## Acknowledgments
