from tensorflow.keras.datasets import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images[:1000]
train_labels = train_labels[:1000]
test_images = test_images[:1000]
test_labels = test_labels[:1000]


conv = Conv3x3(8)  # 28x28x1 -> 26x26x8
pool = MaxPool2()  # 26x26x8 -> 13x13x8
softmax = Softmax(13*13*8, 10)  # 13x13x8 -> 10


def forward(image, label):
    # Completes a forward pass of the CNN and calculates the accuracy and cross-entropy loss.
    # image is a 2d array
    # label is a digit

    # we transform the image from [0,255] to [-0.5,0.5] to make it easier to work with.
    out = conv.forward((image/255)-0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    # calculate cross-entropy loss and accuracy, np.log() is the natural log.
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

def train(im, label, lr=.005):
    # Forward
    out, loss, acc = forward(im, label)

    # Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # Backdrop
    gradient = softmax.backdrop(gradient, lr)
    gradient = pool.backdrop(gradient)
    gradient = conv.backdrop(gradient, lr)

    return loss, acc


print('MNIST CNN initialised')

# Train the CNN for 3 epochs
for epoch in range(3):
    print('--- Epoch %d ---' % (epoch+1))

    # Shuffle the training data
    permutation = np.random.permutation(len(train_images))
    train_images = train_images[permutation]
    train_labels = train_labels[permutation]

    # Train
    loss = 0
    num_correct = 0
    for i, (im, label) in enumerate(zip(train_images, train_labels)):
        if i % 100 == 99:
            print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' % (i + 1, loss / 100, num_correct))
            loss = 0
            num_correct = 0

        l, acc = train(im, label)
        loss += l
        num_correct += acc


# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
num_correct = 0
for im, label in zip(test_images, test_labels):
    l, acc = train(im, label)
    loss += l
    num_correct += acc

num_tests = len(test_images)
print('Test Loss:', loss / num_tests)
print('Test Accuracy:', num_correct / num_tests)

