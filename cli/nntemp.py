import argparse


def main():
    parser = argparse.ArgumentParser(prog='nntemp',
                                     description='Utilitary script to access nntemplate functionalities.')
    nntemp_parser = parser.add_subparsers(dest='command', required=True)
    setup_run(nntemp_parser)
    setup_check(nntemp_parser)

    args = parser.parse_args()
    args.func(args)


def setup_run(parser):
    p = parser.add_parser('run', help='Run a script.')
    p.add_argument('script', help='Path to the script to run.')

    def run(args):
        pass
    p.set_defaults(func=run)


def setup_check(parser):
    p = parser.add_parser('check', help='Check the configuration file.')
    p.add_argument('configuration_file', help='Path to the configuration file to check.')

    def check(args):
        from check import check
        check(args.configuration_file)
    p.set_defaults(func=check)


if __name__ == '__main__':
    main()
