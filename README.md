# Image2schematic

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://makeapullrequest.com) 

Welcome to my Repo! This is my notebook on trying to extract PCB schematics from images using Computer Vision. Any help is greatly appreciated!

I use `OpenCV` (https://opencv.org/) for image detection and modification and `skidl` (https://github.com/devbisme/skidl) for building out a schematic.

**Global SITREP**: Finished working on classes migration.

---------------------------------------------------

There are multiple stages needed for this project:

1. First things first, the raw image needs to be modified so OpenCV better understand it. Something like this is much much better than raw:

    <p align="left"><img src="assets/Example_images/Board_images/Board7.png" alt="assets/Example_images/Board_images/Board7.png" width="400"/></p>

    **SITREP**: currently no progress on that have been achived, any ideas would be greatly appreciated!


2. Using `OpenCV`, we need to understand the pin to pin traces, and assign each pin with it's corrosponding x and y coordinates.

    **SITREP**: Very far into it. `Board8.png` works very good.

3. Using `OpenCV` OR `tesseract`, we need the understand the text on the ICs that are on the board, and then pass that into skidl search function to get the pinout of that Integrated circuit, as well as the schematic symbol of that IC.

    **SITREP**: some what into it, we do know where there are ICs, as well as used skidl queries.

4. Using all this data metioned above, and using skidl, we will make a file with skidl syntax describing the circut, and then output a schematic using `skidl_to_schematic` tool.

    **SITREP**: very close to outputing a test schematic using skidl syntax, a lot of discussion on `skidl` repo.


## Testing

Make sure you have `skidl` and `OpenCV` installed:

```bash
$ pip install skidl
$ pip install opencv-python
```

**Note**: I recommand installing the full package of `OpenCV` as it gives some extra handy modules that might be used in the future. Read more here: https://pypi.org/project/opencv-python/

After you cloned the repository, <ins>first run</ins> `detectingPoints.py`:

```bash
$ python3 detectingPoints.py
```

now you should have `PointsFileFor_Board8.png.txt` file under `output/Files`. This file should include all the coordinates (x,y) of the board electrical points.

Currently there are just two examples: `Board8.png` and `Board9.png`, but feel free to try it on other board images and post your result in the `Discussions` tab.

<p align="left"><img src="assets/Example_images/Board_images/Board8.png" alt="assets/Example_images/Board_images/Board8.png" width="400"/></p>

After that, run `ConnectionFinding.py`:

```bash
$ python3 ConnectionFinding.py
```

If you now look at `PointsFileFor_Board8.png.txt` you should now see every connection from any point: 

```txt
Point: [435,479] connected to: (191,171)
Point: [435,466] connected to: (191,144)
Point: [435,453] connected to: (191,117)
Point: [435,439] connected to: (191,91)
Point: [435,426] connected to: (191,64)
Point: [436,412] connected to: (191,37)
ATtiny841-S | 6/PA7/BIDIRECTIONA | [190,171] connected to: (435,479)
ATtiny841-S | 5/PB2/BIDIRECTIONA | [190,144] connected to: (435,466)
ATtiny841-S | 4/~{RESET}/PB3/BIDIRECTIONA | [190,117] connected to: (435,453)
ATtiny841-S | 3/XTAL2/PB1/BIDIRECTIONA | [190,90] connected to: (435,439)
ATtiny841-S | 2/XTAL1/PB0/BIDIRECTIONA | [190,64] connected to: (435,426)
ATtiny841-S | 1/VCC/POWER-I | [190,37] connected to: (435,406)
```

What we can see are the x,y coordinates of the points, and the x,y coordinates of what point is connected to it. At the bottom we can see what IC is connected to those points and some information about them. (In this example the IC name was hard-coded, IC detection is not ready yet)

This is the baseline to building an entire schematic.

If you encountered any issues during installation or testing of `Image2schematic`, OR if you have any suggestions, please post them in the issues tab.

## Few topics i need help with:

- More than 2 point connection finding - right now I can't detect lines that connect 3 different components at once.
- Skidl_to_schematic algorithm - I can't get the algorithm that take skidl code and output a schematic to work.
- Using yolo or other image classification algorithm to classify a component as either a capacitor, resistor etc.

I want to thank you for reading this and i hope you can help me, thank you!


