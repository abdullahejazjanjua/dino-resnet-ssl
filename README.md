# Branch
- I am re-factoring the codebase, making it more readable.
- Furthermore, I have corrected training mistakes in the codebase.
- I will also train the model and try to replicate the results for ResNet
- On compeletion, this will be the default branch.
- The old (current main default) will remain unchanged



# TODO:

- [x] DinoHead
- [x] Dino Model
- [x] Dino Loss
  - [x] Center update
  - [x] linear warmup of teacher model's temperature (0.04 to 0.07 during the first 30 epochs)
- [x] Augmentations
- [x] Cosine schedule for EMA
- [ ] Dataloader
- [x] Optimizer
  - [x] Linear rampup to base lr: 0.0005 * batchsize/256
  - [x] Use cosine decay after this using cosine schedule
- [x] The weight decay also follows a cosine schedule from 0.04 to 0.4
