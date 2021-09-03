# Quick Links

 - [WPILibPi Releases](https://github.com/wpilibsuite/WPILibPi/releases)
 - [WPILib Vision Documentation](https://docs.wpilib.org/en/stable/docs/software/vision-processing/wpilibpi/using-the-raspberry-pi-for-frc.html)
 - [RobotPy Documentation](https://robotpy.readthedocs.io/en/stable/)
 - [OpenCV Documentation](https://docs.opencv.org/4.5.2/d6/d00/tutorial_py_root.html)
 - [NumPy Documentation](https://numpy.org/doc/)
 - [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)
 - [PSCP](https://it.cornell.edu/managed-servers/transfer-files-using-putty)

# Setup

The Raspberry Pi should be running the [WPILibPi Software](https://github.com/wpilibsuite/WPILibPi/releases). It then only needs to be connected with an ethernet cable to the Rio, and powered from somewhere else. I think we may have used power over ethernet as a backup in addition to the normal cable.

A Raspberry Pi 3 requires 5V & 2.5A to function correctly; The official requirements can be found [here](https://www.raspberrypi.org/documentation/hardware/raspberrypi/power/README.md). Most 5V chargers only use 1A, so double check yours is valid when doing testing, otherwise the Pi will restart for no reason.

If there's connection issues between the Pi and the Rio, then an ethernet hub might be required, we needed one in Deep Space.

# Connecting to the Pi

The Pi is set up by default to connect with using the URL `frcvision.local`. However, we found that this didn't work during the Bot Bash competition, so we switched it to a static IP (which can be done in the web dashboard). Additionally, `frcvision.local` takes a long time to connect on Windows so even if you're using DHCP I'd recommend using the IP address to connect.

To find the IP address, whether on Static or DHCP, you can run the command `hostname -I` on the Pi. You can do this by connecting with a keyboard and signing in with the default `pi` and `raspberry` passwords, or by running the following Python script on the Pi.

```python
import os
print(os.system('hostname -I'))
```

# File Uploads

Files can be uploaded to the Pi in three ways. First, and easiest, is setting the application to Uploaded Python File in the web dashboard. This will take the file uploaded and place it in the home directory of the Pi (renamed as `uploaded.py`) and run it on boot-up.

Secondly, files can be uploaded in the File Upload tab. Any file uploaded here will be placed in the root directory. Additionally, if a zip file is uploaded then it will be extracted to a directory of the same name in the home directory of the Pi. Note that the `runCamera` script will not run any files from these, it will still only call `uploaded.py`. Because of this, I recommend putting a short script in `uploaded.py` that uses a module that you upload as a zip file. You could also just store everything in the same file, but that's a pain, and it if it's separate you can easily test on both your computer and the Pi.

**VERY IMPORTANT NOTE:** when you upload a zip file, the web dashboard does NOT remove and replace the old zip file! You need to do this manually. The easiest way I found to do this was to connect to the Pi manually with [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html) and run the command `rm -r vision`, where vision is the name of whatever zip file you uploaded. This action is permanent-there is no recycling bin. To double-check you're in the right area, `ls` will list all directory content, and you should see the `vision` folder there.

Thirdly, you can manually connect to the Pi using software like [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html) and [PSCP](https://it.cornell.edu/managed-servers/transfer-files-using-putty), but this should only be necessary if you want to edit the Vision module I've set up or if you want to play around with the file structure of the Pi and see how it works.

# Bandwidth Monitoring

By default, FRC enforces a 4Mbs limit, but this can be changed in the
[radio configuration utility](https://docs.wpilib.org/en/stable/docs/zero-to-robot/step-3/radio-programming.html)
if desired for testing or off-season projects.

To monitor bandwidth usage the Windows 10 app Performance Monitor can be used.
Hit the green plus button and select Network Interface, and then delete
everything except total bytes per second and resize the graph axes.

We found that for some reason, we were able to use more bandwidth when we ran two Pis at once: one running solely vision and one solely streaming cameras, as opposed to one Pi that multithreaded the both of them at once.

# Using this Code

The system I've set up here automates a lot of the difficult and tedious parts of the vision system. Mainly, it handles the vision processing step. It does not automate the camera streaming part. I've left this out for a few reasons. First, I never fully understood it, and secondly, the demands vary so much between seasons that it's likely that it would change. For help on how to set it up, check the other two branches of this repository or the example camera streaming file.

There are two main parts: `uploaded.py` and the `vision` module. `uploaded.py` is the code that manages the camera streams and is run on boot-up by the Pi. This can be uploaded using the Uploaded Python File application mode in the web dashboard.

The `vision` module sets up the Processor class that includes all the big features. This is uploaded using the file upload tab as a zip file. To re-upload this, you need to manually remove the old version. The `to_zip.py` file just makes a zip file out of the `vision` directory, I got tired of manually making it.

`Vision Layout.json` is the layout for a [ShuffleBoard](https://docs.wpilib.org/en/stable/docs/software/wpilib-tools/shuffleboard/index.html) dashboard. It works really well, especially at making those last-minute changes at competition. If you change the name of the vision system or add multiple vision systems at once, you will need to recreate this, but it doesn't take that long and is well worth it. Also, changes made here do not save, so use this to find out what values work and to debug, but manually set them in the code. All of the values are stored in the NetworkTables under a single vision name. Each processor you create will have a default name of whatever class the processor is, so in the example it's `ExampleProcessor`. However, this can be changed by setting the variable `PREFIX` in the `__init__` of your processor.

`multiCameraServer.py` is one of the untouched example Python programs that you can download from the Pi, I used it for reference. It works for basic streaming, but code itself is pretty terrible and needlessly complicated.

## What You Need To Do

Your job is to create an `uploaded.py` that has a class that's a child of `TargetProcessor` and manages streaming cameras. I'd recommend reading through this readme, and then at reading all of the processor files and the contour tools file.

You could put your processor class in the module, but I'd recomend putting it in `uploaded.py` so you don't need to delete the old version every time you upload a zip. If you uploaded with the upload python file button it will automatically delete the old one.

Also, I'd say you should read the actual documentation for WPILib up in the quick links. It will give you a broad overview of everything that you need to know. Also, if you'd rather not read this and just want me to give you the run down email me (listed at the end) and I can video call you. My documentation skills are definitely not the best.

## The `vision` Module

The `vision` module contains three main functionalities: configuration, contour tools, and the Processor class. Of these, the Processor class is the largest part.

### Configuration

I've tried to piece together some of the configuration code in the example script and put it in a more Pythonic way. If you need to change this, read through it carefully, cross-reference with the example script, and read the RobotPy documentation (though I found it lacking).

### Contour Tools

Provides a bunch of shortcuts to OpenCV's common contour functions. Read through it, the docstrings give a good description of what they all do. You won't need all of them, but this step will be the bulk of your processing.

### Processor Class

I have three processor classes in the module.

#### `processor.py`

This class doesn't add too much. Mainly it has an abstract method `process_image` you need to define that does the bulk of the work. It also has some preset colors as class variables, and a few methods to draw guidelines, labels, and contours onto an image.

#### `target_processor.py`

This contains much more code, and is probably what you want to use. I have a bunch more utility functions for working with retroreflective tape, and I have it set up so a lot of the work is done, you just need to tune it. There are two abstract methods: `process_contours` and `is_valid`. `process_image` is defined here to get a list of valid contours using what you write in `is_valid`. That list is then passed to `process_contours` where you can do whatever with it.

There's also an example of the solvePnP algorithm called `simple_solve` that you should look at.

This class is set up to work with the ShuffleBoard dashboard.

#### `deep_space_processor.py`

After I wrote this module, I went back and I put the Deep Space Code into it. I never got a chance to test it though, so use it more as a refernce. It's a subclass of `target_processor`.

#### `example_processor.py`

This is also an example I wrote that just detects a normal-sized piece of paper. It's also a subclass of `target_processor`. This is probably a better example than deep space.

#### Synced Values

This is the most important part of the module. The `target_processing` class has a ton of them in the beginning. If you look, they're defined as class variables. However, they function EXACTLY like instance variables. They are class variables in definition only! The important part is that any SyncedValue will automatically update both locally and on NetworkTables. So if you set `output_success` in the code, it will update on the ShuffleBoard dashboard. And if you switch the slider for `view_modified`, getting `view_modified` in the code will return the new value.

Declaring a SyncedValue has two parts: the key, and the default value. The key is where the value will be stored in NetworkTables. In `target_processor.py` you can see I've sorted them into tables and subtables based on their function. I'd recomend doing the same to stay organized.

### Editing the Module

If you don't know how Python modules work, in essence they're a folder of scripts that can be imported into other scripts. If you ever need to add more files or edit existing ones, there are two important things.

If you add a new file, it must include a line under the import statements that reads something like the following. It's a list of everything in the file that should be made public, generally just classes, functions, and constants.

```python
__all__ = ['some_function', 'SOME_CONSTANT', 'SomeClass']
```

Additionally, in the `__init__.py` file you need to include the following. This makes everything that you listed in __all__ available in the module.

```python
from .my_new_file import *
```

If you're importing something from another file in the module to another file in the module, then use the following.

```python
# Only use when writing a file IN the vision module, do not use in uploaded.py!
from .file_in_vision import some_function

# Use inside of uploaded.py. Note that you must have done the __init__ and __all__ steps above if you are importing something you wrote yourself!
from vision import some_function
```

# Camera Calibration

When adjusting the camera settings, I found it much easier to just use the default camera streaming option selectable with the drop-down. The WPILib docs has a good description of what you need to do. They also mention using GREP, you don't need to do that-the Shuffleboard dashboard will do that step for you.

**VERY IMPORTANT NOTE:** Occasionally, the system will crash, and when it does it will delete your camera configuration! Save it as a JSON file after you tune it, and you shouldn't need to touch it again!

# Questions?

Feel free to email me at evangebo1@gmail.com. I'd be happy to help you with any questions, either over email or through a video call.
