# Calculating Light Scattering, Absorption, and Near Field Enhancement in a Multilyered Sphere using Mie Theory

*Code inspired by "Improved recursive algorithm for light scattering by a multilayered sphere" from Applied Optics Yang 2003*

#### Change Log: https://kombateldridge.github.io/

---

## Section 1: Background Information
This walkthrough aims to teach the reader about a recursive algorithm developed by [Wen Yang](https://opg.optica.org/ao/abstract.cfm?uri=ao-42-9-1710) on the topic as well as the addition of a Near Field Enhancement calculation to the system.

This algorithm has been developed to apply Mie Theory [^j] to a multilayered sphere. The algorithm works only for layered spheres, and will not correctly interpret any other morphology. The image below shows an example of concentric layers of spheres as one unit. 
![Multilayer Diagram](/pictures/multilayerimg.png)
### Dielectric File
The user must have text files containing the dielectric/refractive index information for the materials in the multilayered sphere. This file must be placed in the same directory as the code.

The files have to be tab separated lists with no spaces and a header row.

![Dielectric Example](/pictures/diel_img.png)
The script will interpolate between these wavelengths if your dielectric file has asymmetric gaps. The user has the option of compiling dielectric information into one text file or mulitple. 

When using mulitple dielectric files, make sure that every files has its first column as the wavelengths (in nm).

## Section 2: Settings Text File
The script will need information about the system to be analyzed. The mieSettings.txt file is a text file containing the settings of the code. It is *crucial* that the format of the document is not changed. If altered, you'll need to look up how to format the document into a JSON format.

**Setting 1: How many layers?**

Input how many layers in total the system has. This must be an integer greater than 1. If the user wants to run a homomaterial system, just input two or more layers and when asked their radii, let the outermost radius be the true radius and assign the other radii values smaller than the first. Finally, just assign the same dielectric to every shell to get a homomaterial system.

**Setting 2: Dielectric File Location?**

Each item in this array should be the dielectric file for the material of each layer. If the files are not stored in the same directory as the python script, you will need to type the correct path. It is possible to have one dielectric file with multiple materials' dielectric information, just copy the path multiple times in the array.

**Setting 3: What is the radius of each layer?**

Input, starting with the innermost core layer, the outer radius measurement of each layer in **nm**. These can be float values (with decimals).

**Setting 4: What columns of the dielectric file are for each layer?**

Input, starting with the innermost core layer, the column index (where the first column of wavelengths is indexed at 0) for the real and imaginary component of the refractive index of each layer's material. Ex: for the image above, the columns for silver (Ag) are 1 and 2.

**Setting 6: Wavelength Interval**

Dielectric files should contain these wavelengths as data points. Interval between wavelengths will be polynomially interpolated by .py script. Wavelengths should be in nanometers.

## Section 3: Output File

The script will begin to calculate multiple data points for each wavelength. The output file will be a text file containing a header row, columns for each data point, and a row for each wavelength.

![Output File](/pictures/output_img.png)

The outputs are as follows:

- Column 0: Lambda/Wavelength
- Column 1: Qext
- Column 2: Qsca
- Column 3: Qabs
- Column 4: Cext
- Column 5: Csca
- Column 6: Cabs
- Column 7+: Near Field Enhancement for each layer

---
The script for this project can be found [here](https://github.com/kombatEldridge/mieMultilayer/blob/194876399626b4e6be7a885a2d7a52cd65974582/mieMultilayer.py).

[^j]: Describes the scattering of light by a non-absorbing, partially-absorbing, or perfectly conducting sphere.