# build.bat (for Windows)

@echo off
echo Building the Video Labeling App with PyInstaller...

REM Ensure pip is installed
python -m ensurepip --default-pip

REM Install dependencies
pip install pyinstaller
pip install -r requirements.txt

REM Create the 'dist' directory if it doesn't exist
if not exist dist mkdir dist

REM PyInstaller command
REM --onefile: Creates a single executable file.
REM --windowed: Prevents a console window from appearing (for GUI apps).
REM --name: Sets the name of the executable.
REM --add-data: Includes additional files or directories.
REM             Format: "source_path;destination_path_in_bundle"
REM             For Python modules in src, PyInstaller should find them if src is in sys.path.
REM             However, explicitly adding config.py might be safer if it's treated as a data file.
REM             The `.` means put it at the root of the bundle.
REM --hidden-import: Tells PyInstaller to include modules that it might not detect automatically.
pyinstaller ^
    --onefile ^
    --windowed ^
    --name "DrawROIs" ^
    --add-data "src;src" ^
    --add-data "gui;gui" ^
    --hidden-import="tkinter" ^
    --hidden-import="pandas._libs.interval" ^
    --hidden-import="pandas._libs.tslibs.timedeltas" ^
    --hidden-import="pandas._libs.tslibs.timestamps" ^
    --hidden-import="cv2" ^
    --hidden-import="keyboard" ^
    DrawROIs.py

echo Build complete. The executable should be in the 'dist' folder.
pause