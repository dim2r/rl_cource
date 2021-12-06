set PYTHON=python.exe
set env_name=myenv

@echo ----------------------
@%PYTHON% --version
@echo ----------------------

%PYTHON% -m pip install --upgrade pip
%PYTHON% -m pip install --user virtualenv


@echo ---------------------- Creating virtual environment %env_name%   ----------------------
%PYTHON% -m venv %env_name%

set PIP=.\%env_name%\Scripts\pip
set PYTHON=.\%env_name%\Scripts\python.exe


@echo ---------------------- Installing packages into %env_name%   ----------------------
%PIP% install jupyterlab

cd /d %env_name%\Scripts 
.\activate 
cd /d    ..\..\ 

pwd
@%PYTHON% --version
pause

rem %PIP% install wheel
rem %PIP% install six
rem %PIP% install torch torchvision gym gym[atari] numpngw numpy jdc  unrar matplotlib  
%PIP% install torch torchvision gym numpngw numpy pyglet PyYAML jdc    
%PIP% install pybullet
%PIP% install git+https://github.com/benelot/pybullet-gym

rem --------------- BEGIN: CHOOSE ONE OPTION
rem %PIP% install atari_py
rem %PIP% install -f https://github.com/Kojoley/atari-py/releases atari_py
rem --------------- END



%PYTHON% -m atari_py.import_roms rom


@echo ---------------------- starting ----------------------
rem %PYTHON% PG_torch.py

rem .\%env_name%\Scripts\activate


pause