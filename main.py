import argparse
from voxelized_object import VoxelizedObject


def create_parser():
    parser = argparse.ArgumentParser(description='Make it stand')
    parser.add_argument('--mesh_file', type=str, help='mesh input file')
    parser.add_argument('--config_file', type=str, help='configuration file')
    return parser

def parse_config_file(config_file):
    try:
        mesh_rotation = []
        with open(config_file, "r") as f:
            for line in f:
                split_text = line.split()
                if split_text[0] == 'mesh':
                    mesh_file = split_text[1]
                elif split_text[0] == 'mesh_rotation':
                    for rotat_num in split_text[1:]:
                        mesh_rotation.append(float(rotat_num))
                elif split_text[0] == 'flatten_threshold':
                    flatten_threshold = float(split_text[1])
        return mesh_file, mesh_rotation, flatten_threshold
    except IOError:
        print("IOError: {0} could not be opened...".format(config_file))

def main():
    parser = create_parser()
    args = parser.parse_args()
    mesh_file, mesh_rotation, flatten_threshold = parse_config_file(args.config_file)
    VoxelsObject = VoxelizedObject(args.mesh_file, flatten_threshold, mesh_rotation)
    VoxelsObject.prepare_for_balance()
    VoxelsObject.inner_carving(2)
    VoxelsObject.update_after_inner_carving()
    VoxelsObject.save_voxeled_object('inner_carving')
    VoxelsObject.shape_deformation(0.3)
    VoxelsObject.update_after_deformation()
    # VoxelsObject.save_voxeled_object('shape_deformation')


##############################################

if __name__ == '__main__':
    main()
    