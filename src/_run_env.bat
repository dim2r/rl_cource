cd /d myenv\Scripts 
call .\activate.bat
cd /d    ..\..\ 
.\myenv\Scripts\python.exe --version
echo %PROMPT%

.\myenv\Scripts\jupyter-lab.exe   TRPO+PPO_homework.ipynb

exit

rem echo off
rem start "" cmd /k "cd /d env\Scripts & activate & cd /d    ..\..\  "