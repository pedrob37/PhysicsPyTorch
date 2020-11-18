# MICCAI talks
4th October
- iMIMIC: 13:15: Reliable Saliency maps for for Weakly-Supervised Localization of Disease Patterns
          13:30: Explainability for regression CNN in fetal head circumference estimation from ultrasound images
- MLMI: 13:50 - 15:20: Medical Image Segmentation [TBD]

> Reliable Saliency maps for for Weakly-Supervised Localization of Disease Patterns
- Pneumonia detection: Difficult to ascertain why CNN made certain decision
  - Use interpretability to ameliorate this: Highlight regions
  - Localisation overlap not great
- Seek to use weakly-supervised networks to get higher res Saliency maps
  - Encoding backbone + classifier at bottleneck -> Saliency from decoder
  - Weakly supervised: Saliency maps matching of bounding box

3D Brain MRI GAN-based Synthesis Conditioned on Partial Volume Maps
- Use PV maps in GAN for conditioning instead of binary maps
  - Leads to more accurate tissue borders
  - Therefore more suitable to be used in data-scarce applications as more true to form
  - Changes introduced in PV maps reflected in synthetic data
  - Reduction, overall in MSE and MAE

> A Gaussian process model based generative framework for data augmentation of multi-modal 3D image volumes
- Some morphable model + added deformity to produce paired registered image?
  - See poster

> Heterogeneous Virtual Population of Simulated CMR Images for Improving the Generalization of Cardiac Segmentation Algorithms
- Achieving robustness of MRI segmentation algorithms to variable data is challenging
- Simulated images -> Transfer learning
- Adversarial + domain adaptation for future
-

# SASHIMI proceedings: file:///data/Downloads/2020_Book_SimulationAndSynthesisInMedica.pdf

> Contrast Adaptive Tissue Classification by Alternating Segmentation and Synthesis (Dzung L. Pham)
See: https://link.springer.com/chapter/10.1007/978-3-030-59520-3_1
See: https://miccai2020.pathable.co/meetings/virtual/9BwtEBL7bgrR7vzLo

- General spiel about importance of segmentation + poor generalisability to new acquisitions different from training data
  - Segmentation good in DL, but generalisability lacking
- GOAL: Develop segmentation approach that is robust to constrast differences between training data and the input image
- Approach: * Iteratively synthesise new images, without changing labels
            * Contrast Adaptive MEthod for Label IdentiatION (CAMELION)

- Harmonization: * Uses paired/ unpaired training data of different acquisitions to learn an image to image mapping
                 * Sometimes have lackluster data availability
                 * Use segmentations as latent space!

- Method summary: 1. Learn segmentation of one contrast type using standard UNet and Atlas labels
                  2. Create segmentation of different contrast (A) using this approach
                  3. Learn MAPPING between this segmentation and said image (A)
                  4. Use this MAPPING to take original Atlas labels to regenerate original image
                  5. Iteratively update atlas images using this method
                  6. Stop when segmentation changes are small



# New Guotai paper: Uncertainty-Guided Efficient Interactive Refinement of Fetal Brain Segmentation from Stacks of MRI Slices
- Some potentially interesting discussion about this

Deep Generative Model for Synthetic-CT Generation with Uncertainty Predictions: https://miccai2020.pathable.co/meetings/virtual/zZ4J35XWRZ3SEunqx

# Unlearning Scanner Bias for MRI Harmonisation: N K Dinsdale
- Remove scanner information from latent space
- Main task loss + Domain classifier loss + Domain confusion loss


(Yarin Gal) Concrete dropout: How to pick dropout probability
            Has to be tuned
            Increasing data -> More confident -> Reduce dropout probability
            Grid search: Attention plots + ECE + log likelihood etc.


See DART discussion!

0.804 (0.035)
0.715 (0.032)

Demographic information + tubing

...
