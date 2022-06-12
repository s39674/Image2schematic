# Image2schematic - EasyOCR branch

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com) 

This branch is dedicated for EasyOCR integration.

I use `OpenCV` (https://opencv.org/) for image detection and modification and `skidl` (https://github.com/devbisme/skidl) for building out a schematic.

**Global SITREP**: managed to get EasyOCR working with OpenCV. Added a few function to process input, WIP.

---------------------------------------------------


## Testing


EasyOCR, as far as i can tell, only works with `opencv=<4.5.4.60`:

```bash
$ pip install easyocr
# easyocr unnecessary install this package which interfere with opencv-python:
$ pip uninstall opencv-python-headless
# if you have another opencv package, uninstall it and then:
$ pip install opencv-python==4.5.4.60
```

## Few topics i need help with:

- Using yolo or other image classification algorithm to classify a component as either a capacitor, resistor etc.

I want to thank you for reading this and i hope you can help me, thank you!


