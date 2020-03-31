
@ECHO OFF

ECHO Running clang format on this folder...

WHERE clang-format
IF %ERRORLEVEL% NEQ 0 (
  ECHO E: clang-format not found! At least not in the PATH...
) ELSE (
  ECHO clang-format found
  clang-format.exe -i -style=file -verbose ../ImageRanker/src/*.hpp ../ImageRanker/src/*.h ../ImageRanker/src/*.c ../ImageRanker/src/*.cc ../ImageRanker/src/*.cpp

  ECHO Done
)

pause
