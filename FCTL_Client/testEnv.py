from CarlaEnv import CarlaEnv
from carla.tcp import TCPConnectionError
import random
import logging
import time
import sys

if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
    test = CarlaEnv(save_images_to_disk=True)
    connected = False
    while True:
        try:
            if connected:
                test.start_new_episode(0)
                for frame in range(500):
                #     _, r = test.step(steer=random.uniform(-1.0, 1.0), throttle=0.5,
                #                      brake=False, hand_brake=False, reverse=False)
                #     if r > 100:
                #         break

                    test.auto_drive()
            else:
                test.connect()
                connected = True

        except TCPConnectionError as error:
            logging.error(error)
            connected = False
            time.sleep(1)
        except KeyboardInterrupt:
            print('\nCancelled by user. Bye!')
            sys.exit(0)
        except Exception as exception:
            logging.exception(exception)
            sys.exit(1)
