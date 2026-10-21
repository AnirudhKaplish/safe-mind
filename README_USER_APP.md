# SAFE-MIND user app: setup (about 5 minutes)

Three parts, all connected:
 1. Screening engine = your Flask dashboard (pre-filter + baseline + BERT)
 2. Blockchain = consent_server.js (real ganache transactions)
 3. Robot + clinician dashboard = your existing robot window and Meridian

## Step 1. Copy these files into D:\mhd-robot-integration
| From this bundle | To |
|---|---|
| src\dashboard\user_api.py | src\dashboard\user_api.py |
| src\dashboard\user_app.html | src\dashboard\user_app.html |
| blockchain\scripts\consent_server.js | blockchain\scripts\consent_server.js |
| run_user_app.bat | run_user_app.bat (project root) |
| src\dashboard\app.py | src\dashboard\app.py (replaces yours, see step 2) |

blockchain\build\ConsentManager.json must exist (created when you ran deploy_and_demo.js once).

## Step 2. Replace src\dashboard\app.py
The bundle includes your app.py with the connection already added (7 lines at the bottom, above `if __name__`).
Copy it over src\dashboard\app.py (keep a backup of your old one first). Nothing else in it was changed.

## Step 3. Check blockchain packages (once)
    cd /d D:\mhd-robot-integration\blockchain
    npm ls ganache ethers
If missing: npm install ganache ethers

## Step 4. Run
    D:\mhd-robot-integration\run_user_app.bat
Two windows open (blockchain, then dashboard + robot). Wait until the robot says connected.

## Step 5. Open http://127.0.0.1:5000/user
The three chips (Engine, Blockchain, Robot) must be green.

## Step 6. Demo
1. Get started, tick the 4 boxes, Give permission (a real transaction hash appears).
2. Upload data\samples\demo_person_struggling_posts.csv (and comments csv) or paste text.
3. Check my posts: results appear one by one. The robot shows processing, then its face changes.
4. Report: red/amber/green, timeline, themes, urgent note with Tele-MANAS, tx hashes.
5. Revoke permission, try again: blocked.

## Troubleshooting
| Symptom | Fix |
|---|---|
| Engine chip red | Dashboard not running, or app.py was not replaced |
| Blockchain chip red | Blockchain window closed, or port 8546 busy (close old windows) |
| Robot does not react | Robot window must be running and show connected; the page uses the same robot_bus as the dashboard |
| Everything "Too short" | Posts under 6 words are skipped by design |
| Slow | Real BERT needs 1 to 2 minutes for many posts; the live feed shows progress |

## Honest notes
- Tested end to end here with stand-in models and the real consent contract. Test with your real models before the demo.
- The chain is a local ganache chain; it resets when the bridge restarts.
- "Log in with Reddit" is not built (needs OAuth approval); it is future work.
