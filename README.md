# ElectronVision

## Quick Links

 - [WPILibPi Releases](https://github.com/wpilibsuite/WPILibPi/releases)
 - [WPILib Vision Documentation](https://docs.wpilib.org/en/stable/docs/software/vision-processing/wpilibpi/using-the-raspberry-pi-for-frc.html)
 - [RobotPy Documentation](https://robotpy.readthedocs.io/en/stable/)
 - [OpenCV Documentation](https://docs.opencv.org/4.5.2/d6/d00/tutorial_py_root.html)
 - [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)
 - [PSCP](https://it.cornell.edu/managed-servers/transfer-files-using-putty)

## Feature List

 - [x] Easily runnable on computer and Pi
 - [x] Shuffleboard tuning GUI
 - [ ] Intuitive workflow
 - [x] Quick development
 - [x] Does not require extensive knowledge of OpenCV or numpy
 - [ ] Does not require linux knowledge
 - [x] Adaptable to other projects (machine learning and other pipelines)

## Setup

A Raspberry Pi 3 requires 5V & 2.5A to function correctly; The official requirements can be found [here](https://www.raspberrypi.org/documentation/hardware/raspberrypi/power/README.md). Most 5V charges only use 1A, so double check yours is valid, otherwise the Pi will restart for no reason.

If there's connection issues between the Pi and the Rio, then an ethernet hub might be required, we needed one in Deep Space.

## Connecting to the Pi

Normally, you can connect to the Pi by using `frcvision.local/`, or in rare cases `frcvision.lan/`. However, those are really iffy on Windows, so I normally just use the IP address of the Pi.

### Finding the IP Address

If you're connected to the Pi with a keyboard and monitor, use the default username (`pi`) and password  (`raspberry`) to connect and run the command `hostname -I` in the console.

Once you're all set up, I recommend adding a few lines of code to your Python script to print the IP at the start of execution.

```python
import os
print(os.system('hostname -I'))
```

## File Uploads

Files can be uploaded to the Pi in one of two ways. First, the application can be set to uploaded Python file, and a single Python file can be uploaded. This file will be placed in the home directory, renamed to `uploaded.py`, and run by the `runCamera` shell script.

Secondly, files can be uploaded in the File Upload tab. Any file uploaded here will be placed in the root directory. Additionally, if a zip file is uploaded then it will be extracted to a directory of the same name in the root directory. Note that the `runCamera` script will not run any files from these, it will still only call `uploaded.py`. Because of this, I recommend putting a short script in `uploaded.py` that uses a module that you upload as a zip file. You could also just store everything in the same file, but that's a pain, and it if it's separate you can easily test and both your computer and the Pi.

## Testing

### Bandwidth Monitoring

By default, FRC enforces a 4Mbs limit, but this can be changed in the
[radio configuration utility](https://docs.wpilib.org/en/stable/docs/zero-to-robot/step-3/radio-programming.html)
if desired for testing or off-season projects.

To monitor bandwidth usage the Windows 10 app Performance Monitor can be used.
Hit the green plus button and select Network Interface, and then delete
everything except total bytes per second and resize the graph axes.

### Running on a Computer

The code can be run on a computer as well as the Pi. The only thing that needs
to be changed is the method of calling the vision processing class.

## Advanced Usage

### Connecting to the Pi Directly

To connect directly to the Pi, I recommend using the software [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html). To transfer files between your computer and the Pi, I recommend using [PSCP](https://it.cornell.edu/managed-servers/transfer-files-using-putty).

If you're using a static IP for the Pi, then you'll have no issues. However, if you're using DHCP then you need to know the IP.

The Pi runs linux internally, so you should get to know the basics. Check out something like [this quick guide](https://its.temple.edu/linux-quick-reference-guide).

If you ever find yourself wanting to edit some files on the py directly, you can use most Linux text editors but I strongly recommend `nano`, it's much simpler than any of the others and has most basic features.

When you upload a zip file, you need to manually remove the old one. This can be done by running `rm -r vision`, or whatever the name of your directory was.
