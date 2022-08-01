<!--
 Copyright (c) 2022 Innodisk Crop.
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->
# Usage-flip

```
    python3 augCli.py -p <class name> <dataset path> -op <result path> -sp <split ratio> -flip <flip action>
```
Example:
> python3 augCli.py -p D /home/yt/Desktop/DA_testing/D_NG/ F /home/yt/Desktop/DA_testing/F_NG/ -sp 0.8 -op /home/yt/Desktop/DA_testing/train

# Usage-brightness

```
    python3 augCli.py -p <class name> <dataset path> -op <result path> -sp <split ratio> -b <demand of train amount(each class)> <brightness min value> <brightness max value>
```
Example:
> python3 augCli.py -p D /home/yt/Desktop/DA_testing/D_NG/ F /home/yt/Desktop/DA_testing/F_NG/ -op /home/yt/Desktop/DA_testing/train -sp 0.8 -b 40 5 100 

There will be 40 images in your output folder and 5 images in each output_path/val, output_path/test (base on the split ratio).

The images' brightness will be random chose from min value to max value.

