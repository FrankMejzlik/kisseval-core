
@ECHO OFF

ECHO Running clang format on this folder...

WHERE clang-tidy
IF %ERRORLEVEL% NEQ 0 (
  ECHO E: clang-tidy not found! At least not in the PATH...
) ELSE (
  ECHO clang-tidy found
  clang-tidy ../ImageRanker/src/*.hpp ../ImageRanker/src/*.h ../ImageRanker/src/*.c ../ImageRanker/src/*.cc ../ImageRanker/src/*.cpp

  ECHO Done
)

pause
