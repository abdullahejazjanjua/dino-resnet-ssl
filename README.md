# TODO:

- [x] DinoHead
- [x] Dino Model
- [x] Dino Loss
  - [x] Center update
  - [x] linear warmup of teacher model's temperature (0.04 to 0.07 during the first 30 epochs)
- [x] Augmentations
- [] Dataloader
- [] Optimizer
  - [] Linear rampup to base lr: 0.0005 * batchsize/256
  - [] Use cosine decay after this using cosine schedule
- [] The weight decay also follows a cosine schedule from 0.04 to 0.4
  - 