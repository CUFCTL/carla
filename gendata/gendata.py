#!/usr/bin/env python3

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import argparse
import logging

from carla.driving_benchmark import run_driving_benchmark
from carla.driving_benchmark.experiment_suites.fctl_2018 import Fctl2018
from carla.driving_benchmark.experiment_suites.fctl_2018_t import Fctl2018T
from carla.driving_benchmark.experiment_suites.fctl_2018_s import Fctl2018S
from carla.agent.auto_pilot_agent050 import AutoPilotAgent050
from carla.agent.auto_pilot_agent025 import AutoPilotAgent025

if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='verbose',
        help='print some extra status information')
    argparser.add_argument(
        '-db', '--debug',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-c', '--city-name',
           metavar='C',
        default='Town01',
        help='The town that is going to be used on benchmark'
             + '(needs to match active town in server, options: Town01 or Town02)')
    argparser.add_argument(
        '-n', '--log_name',
        metavar='T',
        default='test',
        help='The name of the log file to be created by the benchmark'
    )
    argparser.add_argument(
        '-hn',
        action='store_true',
        default=False,
        help='High Noise Mode'
    )
    argparser.add_argument(
        '-gd',
        action='store_true',
        default=False,
        help='Generate Data'
    )
    argparser.add_argument(
        '-t', '--times',
        metavar='T',
        type=int,
        default=10,
        help='benchmark times')
    argparser.add_argument(
        '-e', '--experiment',
        metavar='E',
        default='CS',
        help='CS: Curve + Straight, C: Curve, S:Straight')
    argparser.add_argument(
        '--continue-experiment',
        action='store_true',
        help='If you want to continue the experiment with the same name'
    )

    args = argparser.parse_args()
    if args.debug:
        log_level = logging.DEBUG
    elif args.verbose:
        log_level = logging.INFO
    else:
        log_level = logging.WARNING

    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)
    logging.info('listening to server %s:%s', args.host, args.port)

    if args.hn:
        agent = AutoPilotAgent050()
    else:
        agent = AutoPilotAgent025()

    if args.experiment == 'C':
        corl = Fctl2018T(args.city_name, args.times)
    elif args.experiment == 'S':
        corl = Fctl2018S(args.city_name, args.times)
    else:
        corl = Fctl2018(args.city_name, args.times)

    # Now actually run the driving_benchmark
    run_driving_benchmark(agent, corl, args.city_name,
                          args.log_name, args.continue_experiment,
                          args.host, args.port, args.gd)
