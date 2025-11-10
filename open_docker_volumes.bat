@echo off
title Open Docker Desktop Volumes Folder

echo ======================================================
echo  Opening Docker Desktop Volumes Folder...
echo ======================================================
echo.
echo  Path: \\wsl.localhost\docker-desktop\mnt\docker-desktop-disk\data\docker\volumes
echo.
echo  If the folder does not open, please ensure that Docker Desktop is currently running.
echo.

start "" "\\wsl.localhost\docker-desktop\mnt\docker-desktop-disk\data\docker\volumes"

timeout /t 5 > nul