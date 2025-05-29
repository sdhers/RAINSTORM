# main.py

import os
import sys
import logging
from tkinter import Tk

# Add the directories to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'gui'))
sys.path.append(os.path.join(script_dir, 'tools'))
sys.path.append(os.path.join(script_dir, 'components'))

from gui.application import VideoProcessorGUI

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(name)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Application starting...")

    root = Tk()
    app = VideoProcessorGUI(root)
    root.mainloop()

    logger.info("Application finished.")