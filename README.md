# Contextual-Bandit-Guided-Adaptive-Attention-Regularization-for-Lung-Cancer-CT-Classification

This is my ongoing research work on AI Safety

# Proposed Method
I propose a Contextual Bandit Guided Adaptive Attention Regularization framework for lung
cancer CT classification.

The CNN will be trained using the original CT images and cancer labels. Anatomical region
masks will be used only as additional supervision for calculating the attention loss. During
training, the model will generate feature maps that will be used to obtain an attention map. The
attention map will then be compared with the anatomical region mask to measure attention
outside the relevant region.

The framework will use a contextual bandit to select the strength of the attention regularization.
The context will initially include Prediction confidence, Classification loss , Attention outside the anatomical region , Model uncertainty , Predicted class

The contextual bandit will choose from several possible actions representing different levels of
attention guidance.For instance: No regularization , Low regularization , Medium regularization , High regularization

The selected action will determine the value of k in the total loss:
Total loss = Classification loss + k * Attention loss
where k is selected adaptively by the contextual bandit.
The bandit will initially be implemented using an interpretable method such as LinUCB or
Thompson Sampling. An initial reward function will combine classification performance and
attention behavior.

# Evaluation
I will compare four approaches:
1. Standard CNN without attention regularization
2. CNN with fixed attention regularization
3. CNN with heuristic adaptive attention regularization
4. CNN with contextual bandit-guided adaptive attention regularization
The models will be evaluated using accuracy, precision, recall, F1-score, ROC-AUC, attention
concentration within the anatomical region, and attention leakage outside the anatomical region.
LIME and Grad-CAM will also be used to analyze model explanations, particularly for correct
and incorrect predictions.

# Research Questions
The study will focus on the following questions:
RQ1: Can anatomically guided attention regularization improve the anatomical relevance of
model attention while maintaining classification performance?
RQ2: Can a contextual bandit learn when and how strongly anatomical attention guidance
should be applied, rather than using the same fixed regularization strategy for every CT image?

# Expected Contribution
This research will extend my previous work from analyzing explanation inconsistencies after
training to actively guiding model attention during training. The main contribution will be an
adaptive framework that learns when and how strongly anatomical attention guidance should be
applied, instead of using the same regularization strength for every CT image.
The study will also investigate whether contextual bandits can provide a simple and interpretable
way to adapt attention regularization according to the characteristics of individual images.

# Previous Work
Poudel, Samarpan, and Vladislav D. Veksler. "Trusting Right Predictions for Wrong Reasons: A
LIME Based Analysis of Deep Learning Interpretability in Lung Cancer Diagnosis." arXiv
preprint arXiv:2606.16036 (2026)
