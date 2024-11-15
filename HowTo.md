# MNIST CNN Training Monitor

This project trains a 4-layer CNN on MNIST dataset with live training monitoring.
mnist_cnn/
├── HowTo.md
├── train.py
├── model.py
├── templates/
│   └── monitor.html
└── static/
    └── style.css

## Requirements 

```
pip install torch torchvision flask numpy matplotlib
```

## How to Run

## 1. Start the training and monitoring server:

```
python train.py
```

## 2. Open your web browser and go to:

```
http://localhost:5000
```

3. You will see the live training progress and loss curves.
4. After training completes, the page will show predictions on 10 random test images.