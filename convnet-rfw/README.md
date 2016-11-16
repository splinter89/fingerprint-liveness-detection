### Convolutional Neural Networks using Random Filter Weights (CNN-RFW)

This is a minimal package that can instantiate and extract features with any CNN model of the family described in [[1]](http://www.coxlab.org/pdfs/fg2011_lfw.pdf).
Its purpose is mainly for the reproduction of that and later results.
If you want to use it as a simple feature extractor based on the model **HT-L3-1st** of [[1]](http://www.coxlab.org/pdfs/fg2011_lfw.pdf), simply install it with

```
python setup.py install
```

and refer to the example code in

```
examples/extract_features.py
```

Otherwise, if you want to hack and extend the package, go with:

```
python setup.py develop
```

In both cases, check if everything is running smoothly with:

```
python setup.py nosetests
```

[[1]](http://www.coxlab.org/pdfs/fg2011_lfw.pdf) Nicolas Pinto and David D. Cox, "Beyond Simple Features: A Large-Scale Feature Search Approach to Unconstrained Face Recognition," in *IEEE Automatic Face and Gesture Recognition*, 2011.
