@echo off
setlocal enabledelayedexpansion

:ASK_SETUP
echo Have you already done the setup? (y/n)
set /p setup_choice=

if /I "!setup_choice!"=="n" (
    echo Running RVC_setup_model.py...
    python RVC_setup_model.py
) else if /I "!setup_choice!"=="y" (
    goto ASK_TRAINING
) else (
    echo Invalid input, please enter y or n.
    goto ASK_SETUP
)

:ASK_TRAINING
echo Do you want to start training? (y/n)
set /p training_choice=

if /I "!training_choice!"=="y" (
    echo Running RVC_train_model.py...
    python RVC_train_model.py
) else if /I "!training_choice!"=="n" (
    goto END
) else (
    echo Invalid input, please enter y or n.
    goto ASK_TRAINING
)

:END
echo Press any key to exit...
pause >nul
