@echo off
setlocal enabledelayedexpansion

cd /d "D:\MetaHack\Round 2 (Finale)\cascade-guard-core-main"

REM Check if node_modules exists
if not exist "node_modules" (
    echo === Step 1: npm ci (installing dependencies) ===
    call npm ci
    set STEP1=!ERRORLEVEL!
) else (
    echo === Step 1: node_modules already exists, skipping npm ci ===
    set STEP1=0
)
echo Exit Code: !STEP1!
echo.

REM Run lint
echo === Step 2: npm run lint ===
call npm run lint
set STEP2=!ERRORLEVEL!
echo Exit Code: !STEP2!
echo.

REM Run test with CI=true
echo === Step 3: npm run test (with CI=true) ===
set CI=true
call npm run test
set STEP3=!ERRORLEVEL!
echo Exit Code: !STEP3!
echo.

REM Run build
echo === Step 4: npm run build ===
call npm run build
set STEP4=!ERRORLEVEL!
echo Exit Code: !STEP4!
echo.

REM Summary
echo === SUMMARY ===
if !STEP1! equ 0 (echo npm ci: PASS ^(Exit: !STEP1!^)) else (echo npm ci: FAIL ^(Exit: !STEP1!^))
if !STEP2! equ 0 (echo npm run lint: PASS ^(Exit: !STEP2!^)) else (echo npm run lint: FAIL ^(Exit: !STEP2!^))
if !STEP3! equ 0 (echo npm run test: PASS ^(Exit: !STEP3!^)) else (echo npm run test: FAIL ^(Exit: !STEP3!^))
if !STEP4! equ 0 (echo npm run build: PASS ^(Exit: !STEP4!^)) else (echo npm run build: FAIL ^(Exit: !STEP4!^))

endlocal
