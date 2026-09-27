# Experiment analysis

Complete this file after running `run_experiments.py`.

1. Which configuration achieved the highest validation accuracy?
2. How did learning rate affect convergence?
3. How did batch size affect accuracy and training time?
4. How did SGD and Adam differ?
5. Compare all four classifiers. Which would you select and why?

6. For CNN and ViT, how did increasing the number of layers from 2 to 5 affect
accuracy, convergence, and training time?

7. For CNN and ViT, compare plain SGD, SGD with momentum, Adam, and AdamW.
What differences did you observe, and which optimizer would you select?


## Answers

### Question 1
The configuration wih the higherst calidation accuracy was the
5 Layer Convolution Nuerla network with
Batch size of 128, learning rate of 0.001 and Adam optimizer with a validation accuracy of approximatey 78%
### Question 2
Reducing the LR from 0.001 to 0.0001 resulted in lower perfomance which  given how few epoch we ran suggested that it resulted in slower convergence. 
comparing the conv_baseline  0.001 and conv_lr_0001 with 0.0001  as well as vit vit_baseline and vit_lr_0001  with smae optimiser, batch size and epoch we see a drop inperfomance as the LR drops
### Question 3 : How did batch size affect accuracy and training time?
Comapring  conv_batch_64 and conv_baseline (64 and 128  batch sizes respectively) we see accuracy of 73.6% and 73.9% and with negligible diffrences as wella s the traingn time  2.9919 and 3.0376 with the 64 batch size taking slightly taking longer to train
### Question 4: How did SGD and Adam differ?

### Question 5:  Compare all four classifiers. Which would you select and why?
As we are perfoming visual tasks id pick the CNN or VIT stylea ccitectures as they ar emore suited for the task compaared to the Linear and Muliti layer perceptron (3 layer) hpwever due to the limited resolution, dataset size and number of epoch the CNN will be the best option as it is more robust and not as subject to the degree of overfitting capalintues with VIT. From the rsults we also see the CNN clearly outperfoming the MLP (47%) and Linear (37%)artchiterctires as they fail to capture spatial featurees with the VIT comping closeset at 58% however due to the downsides wer mention fails to eprfom nearly as good as the CNN with an accuracry of (78%)
### Question 6: For CNN and ViT, how did increasing the number of layers from 2 to 5 affect accuracy, convergence, and training time?

For the ViT Incraesing the numer of layers increased the accuract from (61.2%) to (66.0%) about 5% icrease in perfomance while also increasing the training time per epoch from (3.73s) to (4.73s) as both the forward and backwards pass have more computations to compelte. while being deeper the the 5 layer converges much faster reaching a validation accuracy of  60% muchg faster.
simialrlry for the CNN the 5 layer out perfomrs the 2 layer at 77.9% acurracy and a traning time of 3.09s per epoch compared to the 2 layers 73.9% acurracy and 3.04s per epoch increaing the numbe rof layers increases the training time as well as the accuracy

<!-- conv_baseline,conv,adam,10,10,128,0.001,,2,598,0.82803683,0.73920000,3.0376

conv_layers_5,conv,adam,10,10,128,0.001,,5,598,0.62880056,0.77960000,3.0988



vit_baseline,vit,adam,10,10,128,0.001,,2,598,1.08695029,0.61290000,3.7373
vit_layers_5,vit,adam,10,10,128,0.001,,5,598,0.96223094,0.66000000,4.7302 -->






### Question 7:  For CNN and ViT, compare plain SGD, SGD with momentum, Adam, and AdamW.What differences did you observe, and which optimizer would you select?
1. Plain SGD: Has worst perfomance overall (CNN: 26.6%, ViT: 16.1%)