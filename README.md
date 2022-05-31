# Image2schematic - classes migration

This branch is dedicated to switching to using classes to define each point and each IC on the board.
Using classes will give better control over the link between IC and the points that are connected to it. It also gives me the ability to save the IC/points info in a quicker and easier to work with format.

**Global SITREP**: created `PrintedCircutBoard.py`. Getting ready for merge.

---------------------------------------------------



## Testing

Make sure you have `skidl` and `OpenCV` installed:

```bash
$ pip install skidl
$ pip install opencv-python
```

**Note**: I recommand installing the full package of `OpenCV` as it gives some extra handy modules that might be used in the future. Read more here: https://pypi.org/project/opencv-python/

After you cloned this branch, <ins>first run</ins> `detectingPoints.py`:

```bash
$ python3 detectingPoints.py
```

now you should have `PointsFileFor_Board8.png.txt` file under `output/Files`. This file should include all the coordinates (x,y) of the board electrical points.

Currently there is just one example: `Board8.png`, but feel free to try it on other board images and post your result in the `Discussions` tab.

After that, run `ConnectionFinding.py`:

```bash
$ python3 ConnectionFinding.py
```

If you now look at `PointsFileFor_Board8.png.txt` you should now see every connection from any point. This is the baseline to building an entire schematic.

If you encountered any issues during installation or testing of `Image2schematic`, please post them in the issues tab.

I want to thank you for reading this and i hope you can help me, thank you!


