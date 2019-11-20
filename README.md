# local-binary-patterns-rsna
Local Binary Patterns RSNA

# Install 

```
  $ git clone git@github.com:ai-uff/local-binary-patterns-rsna.git
  $ cd local-binary-patterns-rsna
  $ python3 --version # Python 3.6.7
  $ pip --version # pip 19.3.1 from /usr/local/lib/python3.6/dist-packages/pip (python 3.6)
  $ pip install -r requirements.txt
```
# Instructions

The image directory must have the following format:

```
images
├── testing ----> An image for each label you want to test your model
│   ├── epidural.png ----> label-or-class.png (in this case the label or class is epidural)
│   └── intraparenchymal.png
└── training
    ├── epidural ----> label or class
    │   ├── epidural_01.png ----> add class images here for trainnig
    │   ├── epidural_02.png
    │   ├── epidural_03.png
    │   └── epidural_04.png
    └── intraparenchymal
        ├── intraparenchymal_01.png
        ├── intraparenchymal_02.png
        ├── intraparenchymal_03.png
        └── intraparenchymal_04.png
```

# Run on command line

```
  $ python3 predict.py --training images/training --testing images/testing
```

# Result

A window containing each training image in LBP format and after each learning process a window containing the result for the test images.

# References

1. https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
2. https://en.wikipedia.org/wiki/Local_binary_patterns
3. A great introduction in Portuguese about LBP: http://nca.ufma.br/~geraldo/vc/14.b_lbp.pdf
