@echo OFF

:: Create symlink to data directory
cd ..\ImageRanker
rmdir .\data

cd ..\x64\Release
rmdir .\data

pause