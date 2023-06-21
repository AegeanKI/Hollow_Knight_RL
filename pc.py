import os
import sys
import time

import shutil
import pickle
import cv2
import pyautogui
import torch
import numpy as np

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.

class Monitor:
    def __init__(self, window_title, window_location, window_size):
        self.width, self.height = window_size
        self.left, self.top = window_location
        self.region = (self.left, self.top, self.width, self.height)

        self.window = self.find_window(window_title)
        self.activate_move_to_desired()

    @staticmethod
    def find_window(window_title):
        window = pyautogui.getWindowsWithTitle(window_title)
        assert len(window) == 1, f"found {len(window)} windows called {window_title}"
        window = window[0]
        return window

    @staticmethod
    def activate_window(window):
        try:
            window.activate()
        except Exception:
            window.minimize()
            window.maximize()
            window.restore()

    def capture(self):
        img = pyautogui.screenshot(region=self.region)
        img = np.array(img, dtype=np.float32)
        return img


    def find(self, target_region, target_image):
        return pyautogui.locateOnScreen(target_image,
                                        region=target_region,
                                        confidence=0.95)

    def is_active(self):
        return self.window.isActive

    
    def activate_move_to_desired(self):
        self.activate_window(self.window)
        self.window.resizeTo(self.width, self.height)
        self.window.moveTo(self.left, self.top)


class Keyboard:
    def __init__(self, keys_map):
        self.keys_map = keys_map

    def execute(self, action):
        assert len(action) == len(self.keys_map), f"wrong action size"

        for status, key in zip(action, self.keys_map):
            if status:
                pyautogui.keyDown(key)
            else:
                pyautogui.keyUp(key)

class Logger:
    @staticmethod
    def clear_line():
        sys.stdout.write('\033[2K\033[1G') # erase and go to begining of line

    @staticmethod
    def indent(indent=0):
        print(" " * indent, end='')

class FileAdmin:
    SAFE_SLEEP_TIME = 1
    SAFE_FAIL_SLEEP_TIME = 10

    @staticmethod
    def safe_save_net(net, name, indent=0, quiet=False):
        if not quiet:
            Logger.indent(indent)
            print(f"saving {name}", end='\r')
        done = False
        device = net.device
        while not done:
            try:
                time.sleep(FileAdmin.SAFE_SLEEP_TIME)
                torch.save(net.cpu().state_dict(), name)
                done = True
            except:
                time.sleep(FileAdmin.SAFE_FAIL_SLEEP_TIME)
                if not quiet:
                    print("")
                FileAdmin.safe_remove(name, indent=indent+4, quiet=quiet)
                time.sleep(FileAdmin.SAFE_FAIL_SLEEP_TIME)
        net = net.to(device)
        time.sleep(FileAdmin.SAFE_SLEEP_TIME)
        if not quiet:
            Logger.clear_line()
            Logger.indent(indent)
            print(f"saving {name} completed")

    @staticmethod
    def safe_load_net(net, name, indent=0, quiet=False):
        if not os.path.exists(name):
            return net

        if not quiet:
            Logger.indent(indent)
            print(f"loading {name}", end='\r')
        done = False
        device = net.device
        while not done:
            try:
                time.sleep(FileAdmin.SAFE_SLEEP_TIME)
                net.load_state_dict(torch.load(name))
                done = True
            except:
                time.sleep(FileAdmin.SAFE_FAIL_SLEEP_TIME)
        net = net.to(device)
        time.sleep(FileAdmin.SAFE_SLEEP_TIME)
        if not quiet:
            Logger.clear_line()
            Logger.indent(indent)
            print(f"loading {name} completed")
        return net

    @staticmethod
    def safe_remove(name, indent=0, quiet=False):
        if not os.path.exists(name):
            return

        if not quiet:
            Logger.indent(indent)
            print(f"removing {name}", end='\r')
        done = False
        while not done:
            try:
                time.sleep(FileAdmin.SAFE_SLEEP_TIME)
                os.remove(name)
                done = True
            except:
                time.sleep(FileAdmin.SAFE_FAIL_SLEEP_TIME)
        time.sleep(FileAdmin.SAFE_SLEEP_TIME)
        if not quiet:
            Logger.clear_line()
            Logger.indent(indent)
            print(f"removing {name} completed")

    @staticmethod
    def safe_save(target, name, indent=0, quiet=False):
        if not quiet:
            Logger.indent(indent)
            print(f"saving {name}", end='\r')
        done = False
        while not done:
            try:
                # Logger.indent(indent)
                # print(f"try to save {name}")
                time.sleep(FileAdmin.SAFE_SLEEP_TIME)
                pickle.dump(target, open(name, 'wb'))
                done = True
            except:
                time.sleep(FileAdmin.SAFE_FAIL_SLEEP_TIME)
                if not quiet:
                    print("")
                FileAdmin.safe_remove(name, indent=indent+4, quiet=quiet)
                time.sleep(FileAdmin.SAFE_FAIL_SLEEP_TIME)
        time.sleep(FileAdmin.SAFE_SLEEP_TIME)
        if not quiet:
            Logger.clear_line()
            Logger.indent(indent)
            print(f"saving {name} completed")

    @staticmethod
    def safe_copy(tmp_name, target_name, indent=0, quiet=False):
        if not quiet:
            Logger.indent(indent)
            print(f"copying {tmp_name} to {target_name}", end='\r')
        done = False
        while not done:
            try:
                time.sleep(FileAdmin.SAFE_SLEEP_TIME)
                copy = shutil.copy(tmp_name, target_name)
                done = True
            except:
                time.sleep(FileAdmin.SAFE_FAIL_SLEEP_TIME)
                if not quiet:
                    print("")
                FileAdmin.safe_remove(target_name, indent=indent+4, quiet=quiet)
                time.sleep(FileAdmin.SAFE_FAIL_SLEEP_TIME)
        time.sleep(FileAdmin.SAFE_SLEEP_TIME)
        if not quiet:
            Logger.clear_line()
            Logger.indent(indent)
            print(f"copying {target_name} completed")

    @staticmethod
    def safe_load(target, name, indent=0, quiet=False):
        if not os.path.exists(name):
            return target

        if not quiet:
            Logger.indent(indent)
            print(f"loading {name}", end='\r')
        done = False
        while not done:
            try:
                time.sleep(FileAdmin.SAFE_SLEEP_TIME)
                target = pickle.load(open(name, 'rb'))
                done = True
            except:
                time.sleep(FileAdmin.SAFE_FAIL_SLEEP_TIME)
        time.sleep(FileAdmin.SAFE_SLEEP_TIME)
        if not quiet:
            Logger.clear_line()
            Logger.indent(indent)
            print(f"loading {name} completed")
        return target

