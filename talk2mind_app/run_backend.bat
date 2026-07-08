@echo off
cd talk2mind_app/backend
uvicorn main:app --reload --port 8000