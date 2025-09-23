import argparse
import os
import sys

from speech2motion.service.server import FastAPIServer
from speech2motion.utils.startup import file2dict


def main(args) -> int:
    """Initialize and start the speech2motion server.

    This function sets up the logging directory, loads the configuration
    from the specified file, initializes the appropriate server type,
    and starts the server.

    Args:
        args: Parsed command line arguments containing config_path.

    Returns:
        int: Exit code (0 for success).
    """
    if not os.path.exists('logs'):
        os.makedirs('logs')
    startup_config = file2dict(args.config_path)
    # init server
    cls_name = startup_config.pop('type')
    if cls_name == 'FastAPIServer':
        server = FastAPIServer(**startup_config)
    else:
        raise ValueError(f'Invalid server type: {cls_name}')
    server.run()
    return 0


def setup_parser():
    """Set up command line argument parser for the speech2motion server.

    Creates an argument parser with configuration for the server startup,
    including the path to the configuration file.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser('Start the backed program.')
    # server args
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to the config file, which contains the server info.',
        default='configs/local.py')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = setup_parser()
    ret_val = main(args)
    sys.exit(ret_val)
