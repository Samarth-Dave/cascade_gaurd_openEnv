# Questions Harshit Video Log

Keep appending each question and a short answer here.

## 2026-04-25

1. Q: I am getting path, FastAPI, and npm command errors in PowerShell. Can you fix this?
   A: Yes. The path errors came from using a non-existent folder; the correct repo is D:\HACKATHON\cascade_gaurd_openEnv. FastAPI was missing in the active venv, so backend dependencies were installed. Node.js/npm were not installed, so Node LTS was installed and UI dependencies were restored. A machine policy also blocked optional Gradio/Pandas imports used by OpenEnv web UI, so the optional import guard was patched locally in the venv to allow backend API startup. Final result: backend starts on port 8000 and frontend starts on port 5173.
   Beneficial for project: Yes - it unblocks local demo execution for both backend and frontend and avoids environment-related demo failures.

2. Q: Why do I get WinError 10048 when running python server/app.py, and what command fixes it?
   A: WinError 10048 means port 8000 is already in use, usually by an already running backend instance. Free the port by stopping the process bound to 8000, then start the server again. If you want to keep the old process running, start this server on a different port instead.
   Beneficial for project: Yes - it prevents backend launch failures during demos and local testing.

3. Q: npm is not recognized in PowerShell even after winget says Node LTS is installed. How do I fix and run the UI?
   A: Node was installed at C:\Program Files\nodejs, but that path was missing from the terminal PATH in the current VS Code session. The fix was to add C:\Program Files\nodejs to PATH (user PATH already updated), then prepend it in-session with $env:Path = 'C:\Program Files\nodejs;' + $env:Path. After that, npm -v worked, npm install completed, and npm run dev started Vite at http://127.0.0.1:5173.
   Beneficial for project: Yes - it restores frontend dependency install and local dev server startup, unblocking UI development and demo flow.
