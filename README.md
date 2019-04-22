# Deep-learning-hw3

## Question 1

## Question 2
VAE model in``` VAE.py```

VAE training and log-likelihood estimate in ```vae_train.py```.
To reproduce all the results run
```
python vae_train.py
```
Method to train and evaluate the trained model on validation and test set:
```
main(train_loader, valid_loader, test_loader, n_epochs, device, lr=3e-4)
```
Method for one epoch of training is:
```
train(model, optimizer, epoch, train_loader, device)
```
Method to evaluate the trained model is:
```
test(model, epoch, valid_loader, device, split="Valid")
```
Method to sample K z for q(z|x):
```
generate_K_samples(mu, logvar, K)
```
Method for calculating log-likelihodd on a mini-batch using importance sampling:
```
log_px = importance_sampling(model, mini_batch_x, Z)
```
Method for estimating log-likelihodd on valid or test set using importance sampling:
```
estimate_log_likelihood(data_loader, device, split="valid")
```
