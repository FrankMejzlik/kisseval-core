@echo OFF

:: Create symlink to data directory
cd ..\ImageRanker
rmdir .\data
mklink /J .\data c:\Users\devwe\data\

cd ..\x64\Release
mklink /J .\data c:\Users\devwe\data\

pause