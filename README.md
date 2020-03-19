# tabrecognition
Basic Optical Guitar Tablature Recognition (L248 Mini-Project)

## Usage
Python 3.6
- run `mkdir saved_model` in this directory
- to train and save models, run:
  - `python train_model.py`
  - `python train_model.py -l`
- to convert an image file `fprefix.jpg`:
  - run `python convert_tab.py img/fprefix.jpg [-l]`
  - validation output will appear as `fprefix_validation_[True/False].png`
