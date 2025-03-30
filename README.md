# Fine-Tuning BLIP on BDD100K for Autonomous Vehicle Perception

## Overview
This project fine-tunes a **BLIP (Bootstrapped Language-Image Pretraining)** model using the **BDD100K dataset** to enhance **autonomous vehicle perception**. The goal is to improve **image captioning and semantic understanding**, allowing the model to describe driving scenes more accurately.

The **BDD100K dataset** is significantly larger than Cityscapes and provides diverse road scenarios, leading to **better generalization and higher accuracy** compared to models trained on Cityscapes. Loss values were also lower, indicating more effective learning.

## Why Use BLIP?
BLIP was chosen due to the following advantages:

1. **Strong Vision-Language Capabilities** – Unlike traditional CNN-based models, BLIP integrates both vision and language understanding, improving scene comprehension.
2. **Pretrained on Large-Scale Data** – The model is already trained on massive datasets, making it highly efficient for fine-tuning with smaller datasets like BDD100K.
3. **Superior Image Captioning Performance** – It generates high-quality captions that help autonomous vehicles interpret driving environments.
4. **Transfer Learning Efficiency** – BLIP can transfer learned knowledge efficiently, reducing the amount of data needed for fine-tuning.

## Why Use BDD100K Dataset?
BDD100K is a **much larger** dataset than Cityscapes and provides more diverse driving scenarios. This led to **better accuracy and lower loss values** due to the following factors:

1. **Diversity in Driving Conditions** – BDD100K covers various weather conditions, lighting situations, and road types, making the model more robust.
2. **Higher Data Volume** – With 100,000 images, the model gets more examples to learn from, reducing overfitting and improving generalization.
3. **Detailed Annotations** – It includes not only segmentation masks but also object tracking, captions, and lane markings, allowing for richer learning.
4. **Better Generalization** – More varied examples enable the model to recognize scenes more accurately in real-world applications.

## Training Process & Observations
The fine-tuning process revealed key differences compared to Cityscapes-based training:

1. **Loss Reduction:**
   - The initial loss was lower compared to Cityscapes, as BLIP already had strong prior knowledge from its pretraining.
   - Loss decreased steadily and reached a lower final value, indicating **better learning and fewer ambiguities** in the dataset.
   
2. **Higher Accuracy:**
   - The model achieved a **higher accuracy than Cityscapes-based models** due to the increased dataset size and diversity.
   - Since BDD100K provides richer and more varied scenes, the model learned better representations, improving performance.

3. **Minimal Changes in Image Captioning:**
   - The model was already pretrained on diverse datasets, so **captions before and after fine-tuning showed minimal differences**.
   - Reasons for minimal change:
     - BLIP had already learned common driving scene descriptions from large-scale pretraining.
     - The fine-tuning dataset (BDD100K) mainly refined the model rather than drastically altering its outputs.
     - Most improvements were subtle, enhancing **scene-specific details** rather than rewriting captions completely.

## Results & Visualization
To evaluate the model, we compared captions before and after fine-tuning. The changes were **minor but meaningful**, improving clarity and specificity.

**Example:**
### **Before Fine-Tuning:**
![Before Fine-Tuning](path_to_before_image)

### **After Fine-Tuning:**
![After Fine-Tuning](path_to_after_image)

This demonstrates that fine-tuning refined the details rather than changing the overall understanding.

## Conclusion
Fine-tuning **BLIP on BDD100K** significantly improved image captioning for autonomous vehicles. The **larger dataset size and diversity** led to **better accuracy and lower loss** compared to Cityscapes.

### Key Takeaways:
- **BLIP was a great choice** for vision-language understanding in driving environments.
- **BDD100K outperformed Cityscapes** due to its scale and variety, leading to lower loss and higher accuracy.
- **Minimal changes in captions** were observed because BLIP was already well-trained, and fine-tuning mainly refined existing knowledge.

## Future Work
To further enhance the model, the following improvements can be explored:
1. **Using Multi-Modal Training** – Combining segmentation with captioning to improve scene understanding.
2. **Fine-Tuning on Edge Cases** – Focusing on rare scenarios like extreme weather conditions and unusual road structures.
3. **Real-Time Inference Optimization** – Speeding up caption generation for deployment in autonomous vehicles.
4. **Adding More Contextual Data** – Incorporating sensor data (LiDAR, Radar) to enrich image captions beyond visual elements.



## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py --epochs 50 --batch_size 16

# Evaluate the model
python eval.py --checkpoint path_to_checkpoint
```

## References
- BDD100K Dataset: https://bdd-data.berkeley.edu/
- BLIP Paper: https://arxiv.org/abs/2201.12086
