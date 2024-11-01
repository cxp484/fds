@echo off
set LIB_TAG=v6.7.0

::*** library and tag name are the same

set LIB_DIR=%LIB_TAG%

::*** placehoder for parsing options

call :getopts %*
if %stopscript% == 1 exit /b

::*** make sure cmake and gcc are installed

set abort=0
set buildstatus=
call :is_file_installed cmake || set abort=1
call :is_file_installed gcc   || set abort=1
if %abort% == 1 exit /b

set CURDIR=%CD%

::*** define root directory where fds repo and libs directories are located

set FIREMODELS=..\..\..\..
cd %FIREMODELS%
set FIREMODELS=%CD%
cd %CURDIR%

::*** if sundials library directory exists exit and use it

set INSTALLDIR=%FIREMODELS%\libs\sundials\%LIB_DIR%
if not exist %INSTALLDIR% goto endif1
  set SUNDIALS_HOME=%INSTALLDIR%
  set buildstatus=prebuilt
  goto eof
:endif1

::*** if directory pointed to by SUNDIALS_HOME exists exit and use it

if "x%SUNDIALS_HOME%" == "x" goto endif2
  if not exist %SUNDIALS_HOME%  goto endif2
    set buildstatus=prebuilt
    goto eof
:endif2

::*** if sundials repo does not exist exit and build fds without it

set LIB_REPO=%FIREMODELS%\sundials
if exist %LIB_REPO% goto endif3
  set SUNDIALS_HOME=
  set buildstatus=norepo
  goto eof
:endif3

::*** if we've gotten this far the prebuilt libraries do not exist, the repo does exist so build the sundials library

cd %LIB_REPO%

set buildstatus=build
echo.
echo ----------------------------------------------------------
echo ----------------------------------------------------------
echo building Sundials library version %LIB_TAG%
echo ----------------------------------------------------------
echo ----------------------------------------------------------
echo.

echo.
echo ----------------------------------------------------------
echo ----------------------------------------------------------
echo setting up Intel compilers
echo ----------------------------------------------------------
echo ----------------------------------------------------------
echo.
call %FIREMODELS%\fds\Build\Scripts\setup_intel_compilers.bat

git checkout %LIB_TAG%

echo.
echo ----------------------------------------------------------
echo ----------------------------------------------------------
echo cleaning sundials repo
echo ----------------------------------------------------------
echo ----------------------------------------------------------
echo.

cd %LIB_REPO%
set BUILDDIR=%LIB_REPO%\BUILDDIR
git clean -dxf

mkdir %BUILDDIR%
cd %BUILDDIR%

::*** configure sundials

echo.
echo ----------------------------------------------------------
echo ----------------------------------------------------------
echo configuring sundials version %SUNDIALSTAG%
echo ----------------------------------------------------------
echo ----------------------------------------------------------
echo.

cmake ..\  ^
-G "MinGW Makefiles" ^
-DCMAKE_INSTALL_PREFIX="%INSTALLDIR%" ^
-DEXAMPLES_INSTALL_PATH="%INSTALLDIR%\examples" ^
-DBUILD_FORTRAN_MODULE_INTERFACE=ON ^
-DCMAKE_C_COMPILER=icx ^
-DCMAKE_Fortran_COMPILER=ifort ^
-DEXAMPLES_ENABLE_C=OFF ^
-DEXAMPLES_ENABLE_CXX=OFF ^
-DEXAMPLES_ENABLE_F2003=OFF ^
-DENABLE_OPENMP=ON ^
-DBUILD_SHARED_LIBS=OFF ^
-DCMAKE_INSTALL_LIBDIR="lib" ^
-DCMAKE_C_FLAGS_RELEASE="${CMAKE_C_FLAGS_RELEASE} /MT" ^
-DCMAKE_C_FLAGS_DEBUG="${CMAKE_C_FLAGS_DEBUG} /MTd"

::*** build and install sundials

echo.
echo ----------------------------------------------------------
echo ----------------------------------------------------------
echo building sundials version %LIB_TAG%
echo ----------------------------------------------------------
echo ----------------------------------------------------------
echo.
call make 

echo.
echo ----------------------------------------------------------
echo ----------------------------------------------------------
echo installing sundials version %LIB_TAG% in %INSTALLDIR%
echo ----------------------------------------------------------
echo ----------------------------------------------------------
echo.
call make install

echo.
echo ----------------------------------------------------------
echo ----------------------------------------------------------
echo setting SUNDIALS_HOME environment variable to %INSTALLDIR%
set SUNDIALS_HOME=%INSTALLDIR%
echo.

echo sundials version %LIB_TAG% installed in %INSTALLDIR%
echo ----------------------------------------------------------
echo ----------------------------------------------------------
echo.

cd %CURDIR%

goto eof

:: -------------------------------------------------------------
:getopts
:: -------------------------------------------------------------
 set stopscript=0
 if (%1)==() exit /b
 set valid=0
 set arg=%1
 if /I "%1" EQU "-help" (
   call :usage
   set stopscript=1
   exit /b
 )
 shift
 if %valid% == 0 (
   echo.
   echo ***Error: the input argument %arg% is invalid
   echo.
   echo Usage:
   call :usage
   set stopscript=1
   exit /b
 )
if not (%1)==() goto getopts
exit /b

:: -------------------------------------------------------------
:is_file_installed
:: -------------------------------------------------------------

  set program=%1
  where %program% 1> installed_error.txt 2>&1
  type installed_error.txt | find /i /c "Could not find" > installed_error_count.txt
  set /p nothave=<installed_error_count.txt
  erase installed_error_count.txt installed_error.txt
  if %nothave% == 1 (
    echo "***Fatal error: %program% not present"
    exit /b 1
  )
  exit /b 0

:: -------------------------------------------------------------
:have_program
:: -------------------------------------------------------------
:: same as is_file_installed except does not abort script if program is not insstalled

  set program=%1
  where %program% 1> installed_error.txt 2>&1
  type installed_error.txt | find /i /c "Could not find" > installed_error_count.txt
  set /p nothave=<installed_error_count.txt
  erase installed_error_count.txt installed_error.txt
  if %nothave% == 1 (
    exit /b 1
  )
  exit /b 0

:: -------------------------------------------------------------
:usage  
:: -------------------------------------------------------------
echo build sundials
echo. 
echo -help           - display this message
exit /b

:eof
echo.
if "%buildstatus%" == "norepo"   echo Sundials library not built, The sundials git repo does not exist
if "%buildstatus%" == "prebuilt" echo Sundials library not built. It exists in %SUNDIALS_HOME%
