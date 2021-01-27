# make_it_stand

usage: main.py [-h] [--mesh_file MESH_FILE] [--config_file CONFIG_FILE]

Make it stand

optional arguments:
  -h, --help            show this help message and exit
  --mesh_file MESH_FILE
                        mesh input file
  --config_file CONFIG_FILE
                        configuration file
						
In the library make-it-stand we have an objects folders that contains 3 sub-folders for each object. 
this sub-folder contains .obj files and config.mis file.
obj file contains the mesh file and .mis containing the configuration file for the mesh file.

the mis file contains:
1. name of the mesh file it refers to
2. mesh rotation to compute the gravity
3. flatten threshold to compute the support vector height

The 3 folder objects are:
1. Horse - containing horse.obj and config.mis file
2. Dinosaure - containing dino.obj and config.mis file
3. Bird - bird.obj and config.mis file

main.py is the main file to run make-it-stand project
main.py requires 2 files: .obj file and .mis file
For example:
python main.py --mesh_file ./Horse/horse.obj --config_file ./Horse/horse.mis

We support only one iteration of inner carving and shape deformation although it is simple the add more iterations

Our output after each step is an exported .obj file in runResults folder that represent the result object
This folder should contain - 
1. object_after_inner_carving.obj
2. object_after_shape_deformation.obj

