# FRC Team 2619 Vision
This is the vision program the we used for the 2020 Infinite Recharge season.

## FRC Rasberry Pi Dashboard
You can view the official documentation [here](https://wpilib.screenstepslive.com/s/currentCS/m/85074/l/1027798-the-raspberry-pi-frc-console).

### Powering the Pi
Make sure you have enough power supplied to the Pi, or else it will randomly crash and cause many issues. The official requirements are found [here](https://www.raspberrypi.org/documentation/hardware/raspberrypi/power/README.md), but in short for the Raspberry Pi 3 it's 5V & 2.5A.

### Connecting to the Pi
When connecting over ethernet, you first have to change your ethernet connection settings. Right click on the wifi icon and select `Open Network & Internet Settings`. Then select `Change Adapter Options` and double click into the ethernet port that you're using. Select `Properties` and then select the properties of `Internet Protocol Version 4 (TCP/IPv4)`. Change the IP address to `10.26.19.5` and the subnet mask to `255.0.0.0`. The other settings shouldn't need to be changed. You can also select the option entitled `DHCP` which doesn't require you to change any ethernet settings, but while DHCP worked for most competitions it broke for our off-season one.

To connect to the dashboard go to `frcvision.local/` or if that doesn't work the IP address of the Pi, so `10.26.19.12/`.

#### Setting static IP address on Pi
IPv4 Address: `10.26.19.12`
Subnet Mask: `255.255.255.0`
Gateway: `10.26.19.1`
DNS Server: `10.26.19.1`

#### If you can't connect to the Pi
If you have an issue connecting to the Pi with a static IP, you can find the IP of the Pi by connecting to it with a keyboard and monitor and running the command `sudo ifconfig`. The IP of the Pi is listed as `inet` under `eth0` (make sure the Pi is still connected to your computer via ethernet if it doesn't appear). Then set the IP address of the computer's ethernet to `XXX.XXX.XXX.5` and the subnet mask to `255.0.0.0`.

### Saving camera parameters
I've had a lot of trouble with saving the camera parameters. Properties such as gain and exposure wouldn't change correctly, it would take severl tries and they wouldn't change on their own. I don't know how I got it working, it just works. My advice: whenever you change the camera parameters keep a backup of the previous settings in case things break. Also if you still have trouble try change the parametes ddirectly from the json file and not hitting their save to json file option. Another note, if you try and boot the pi but it gives you an error that says something about a json file not being found, just resave the camera settings and it should fix it.

## Machine Learning

We might do machine learning, we might not. The docs for it are [here](https://docs.wpilib.org/en/latest/docs/software/examples-tutorials/machine-learning/index.html). The docs direct you to the site [Supervisely](https://supervise.ly/) to get the dataset. The images might be useful as data to get a ball filter as another option than machine learning. An excerpt from some of this downloaded data that could be used for that is in the folder titled `Power Cell Images`.
