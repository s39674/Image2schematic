# Image2schematic

Welcome to my Repo! This is my notebook on trying to extract pcb schematics from images using computer vision. Any help is greatly appreciated!

I use `OpenCV` for image detection and modification and `skidl` for building out a schematic.  

SITREP: Currently Created a github account and made this Readme file.

---------------------------------------------------

There are multiple stages needed for this project:

1. First things first, the raw image needs to be modified so OpenCV better understand it. something like this is much much better than raw:
{image here}

    SITREP: currently no progress on that have been achived.


2. Using OpenCV, we need to understand pin to pin traces, and assign each pin with it's corrosponding x and y cords.

    SITREP: very far into it, need to upload the files to github.


3. Using OpenCV, we need the understand the text on the ICs on the board, and then pass that into skidl search function the get the pinout of that Integrated circuit, as well as the schematic symbol of that IC.

    SITREP: some what into it, we do know where the are ICs.

4. Using all this data metioned above, and using Skidl, we will make a file with skidl syntax descirbing the circut, and then output a schematic.

    SITREP: very close to outputing a test schematic using skidl syntax.