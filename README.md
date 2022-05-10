# Image2schematic-amirs test version

Welcome to my Repo! This is my notebook on trying to extract pcb schematics from images using computer vision. Any help is greatly appreciated!

I use `OpenCV` (https://opencv.org/) for image detection and modification and `skidl` (https://github.com/devbisme/skidl) for building out a schematic.  

**Global SITREP**: Currently Working on getting skidl to schematic ready for use.

---------------------------------------------------

There are multiple stages needed for this project:

1. First things first, the raw image needs to be modified so OpenCV better understand it. Something like this is much much better than raw:

    <p align="left"><img src="assets/Example_images/Board_images/Board7.png" alt="assets/Example_images/Board_images/Board7.png" width="400"/></p>

    **SITREP**: currently no progress on that have been achived, any ideas would be greatly appreciated!


2. Using OpenCV, we need to understand the pin to pin traces, and assign each pin with it's corrosponding x and y cords.

    **SITREP**: very far into it, need to upload the files to github.

3. Using OpenCV, we need the understand the text on the ICs on the board, and then pass that into skidl search function the get the pinout of that Integrated circuit, as well as the schematic symbol of that IC.

    **SITREP**: some what into it, we do know where there are ICs, as well as used skidl queries.

4. Using all this data metioned above, and using skidl, we will make a file with skidl syntax descirbing the circut, and then output a schematic.

    **SITREP**: very close to outputing a test schematic using skidl syntax, lots of discussion on his repo.
