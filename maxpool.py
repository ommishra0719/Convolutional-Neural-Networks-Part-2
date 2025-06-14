import numpy as np


class MaxPool2:
    # A max pooling layer using pool size of 2

    def __init__(self):
        self.last_input = None

    def iterate_regions(self, image):
        # Generating non-overlapping 2x2 image regions to pool over
        # image is a 2d numpy array
        h, w, _ = image.shape
        new_h = h//2
        new_w = w//2

        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield im_region, i, j

    def forward(self, input):
        # Performs a forward pass of the maxpool layer using the given input
        # Returns a 3d array with dimensions (h/2, w/2, num_filters)
        self.last_input = input
        
        h, w, num_filters = input.shape
        output = np.zeros((h//2, w//2, num_filters))

        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(im_region, axis=(0, 1))

        return output

    def backdrop(self, d_L_d_out):
        d_L_d_input = np.zeros(self.last_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0,1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If this pixel was the max value, copy the gradient to it.
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input
