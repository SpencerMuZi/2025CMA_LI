# 2025CMA_LI

## Confidence Memory Attack
1. gray-box attack
    + Attackers can obtain the training data of the attacked model, but do not know the structure and parameters of the model
2. maintains attack effectiveness on par with white-box method while possessing
superior stealthiness
3. CCA firstly generate malicious data and augment it to the model’s training dataset. Then CCA designs a malicious loss function, which could enforce the model to encode the output confidences with the private data to be stolen. Finally, the adversary use the output confidences of the trained model to recover the stolen data. CCA can stealthily launch attacks in gray-box environments without utilization of the model parameters.