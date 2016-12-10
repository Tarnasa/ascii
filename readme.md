# Module Requirements
This project was developed in python2 and requires the following modules:
- scipy
- numpy
- scikit-image
- scikit-learn

These can be installed using `pip install <module>` for each respective module.

# File Dependencies
In order to run, a set of rendered ASCII characters in the desired font must be present in the out/ directory. These can be generated in different fonts using `text.py`, and these are used as the training data for the nearest-neighbors and single-sample SVM approaches.
Additionally, for the regular SVM to run, the training.dat file must be present in the base directory. This can be found in training.zip, or generated using `generate_data.py`.

# To Run
`python convert.py <image name>` will convert the image file provided as a command line argument into ASCII art. The output will be directly displayed to the console along with timings using each of our three algorithms.